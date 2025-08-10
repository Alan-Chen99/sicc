from contextlib import contextmanager
from dataclasses import dataclass

import networkx as nx
from rich import print as print  # autoflake: skip
from rich.pretty import pretty_repr
from rich.text import Text

from .._core import FORMAT_ANNOTATE
from .._core import AlwaysUnpack
from .._core import BoundInstr
from .._core import MVar
from .._core import NeverUnpack
from .._core import ReadMVar
from .._core import Undef
from .._core import UnpackPolicy
from .._core import WriteMVar
from .._instructions import Move
from .._utils import cast_unchecked_val
from .basic import get_index
from .control_flow import CfgNode
from .control_flow import build_control_flow_graph
from .control_flow import external
from .utils import LoopingTransform
from .utils import Transform
from .utils import TransformCtx


@dataclass
class LifetimeResPerInstr:
    # only constructed if cur is not a WriteMvar

    # the mvar value BEFORE cur may be used latter at ...
    # note: this means:
    # includes cur, if cur is a ReadMVar
    # if cur is WriteMvar, this would be empty by definition;
    # note that cur would then not be in reachable and we dont construct the object in this case
    possible_uses: list[BoundInstr[ReadMVar]]

    # the value of the mvar BEFORE cur may come from ...
    # if possible uses is empty, this is empty too, even if it may have a value
    # a def -> external -> cur counts
    possible_defs: list[BoundInstr[WriteMVar]]

    # is there a path external -> cur touching no defs?
    possible_undef: bool = False


@dataclass
class MvarLifetimeRes:
    v: MVar
    #: set of instr where the value of v at the instr may get used latter
    #: does not include any def or uses of v
    reachable: dict[CfgNode, LifetimeResPerInstr]

    def annotate(self, instr: BoundInstr) -> Text:
        if info := self.reachable.get(instr):
            return Text(pretty_repr(info), "ic10.comment")
        return Text()

    @contextmanager
    def with_anno(self):
        with FORMAT_ANNOTATE.bind(self.annotate):
            yield


def support_mvar_analysis(
    ctx: TransformCtx, v_: MVar, unpack: UnpackPolicy = NeverUnpack()
) -> bool:
    index = get_index.call_cached(ctx, unpack)
    v = index.mvars[v_]

    if not v.private:
        return False

    # not sufficiently expanded
    if not all(x.isinst(WriteMVar) for x in v.defs):
        return False
    if not all(x.isinst(ReadMVar) for x in v.uses):
        return False

    return True


def compute_mvar_lifetime(
    ctx: TransformCtx, v_: MVar, unpack: UnpackPolicy = NeverUnpack()
) -> MvarLifetimeRes:
    graph = build_control_flow_graph.call_cached(ctx, out_unpack=unpack)
    index = get_index.call_cached(ctx, unpack)

    assert support_mvar_analysis(ctx, v_, unpack=unpack)
    v = index.mvars[v_]

    defs = [d.check_type(WriteMVar) for d in v.defs]
    uses = [u.check_type(ReadMVar) for u in v.uses]

    ########################################
    # get the subgraph where a value set to v
    # may get used
    # "ancestors" call wraps around "external"

    hide_defs = cast_unchecked_val(graph)(
        nx.restricted_view(graph, defs, []),  # pyright: ignore[reportUnknownMemberType]
    )
    res: dict[CfgNode, LifetimeResPerInstr] = {}

    def get(x: CfgNode) -> LifetimeResPerInstr:
        if ans := res.get(x):
            return ans
        ans = LifetimeResPerInstr([], [])
        res[x] = ans
        return ans

    for use in uses:
        get(use).possible_uses.append(use)
        for anc in nx.ancestors(hide_defs, use):  # pyright: ignore[reportUnknownMemberType]
            get(anc).possible_uses.append(use)

    ########################################

    # does not include any defs
    # include external, if it can reach any use without reaching a def first
    reachable = set(res)

    for d in defs:
        subgraph = graph.subgraph(reachable | {d})
        des = nx.descendants(subgraph, d)  # pyright: ignore[reportUnknownMemberType]1
        for x in des:
            res[x].possible_defs.append(d)

    subgraph = graph.subgraph(reachable | {external})
    des = nx.descendants(subgraph, external)  # pyright: ignore[reportUnknownMemberType]1
    for x in des:
        res[x].possible_undef = True

    if external in reachable:
        res[external].possible_undef = True

    ########################################

    # for use in uses:
    #     if len(res[use].possible_defs) == 0:
    #         use.debug.error("use of mvar that can never be defined").throw()

    # sort since iteration may not be deterministic
    return MvarLifetimeRes(v.v, dict(sorted(res.items())))


@LoopingTransform
def elim_mvars_read_writes(ctx: TransformCtx, unpack: UnpackPolicy = AlwaysUnpack()) -> bool:
    """
    # (1) optimize mvar reads that can only come from a single mvar write
    # (2) remove mvar write that can never b read
    """
    f = ctx.frag
    graph = build_control_flow_graph.call_cached(ctx, out_unpack=unpack)
    index = get_index.call_cached(ctx, unpack)

    for v in index.mvars.values():
        if not support_mvar_analysis(ctx, v.v, unpack):
            continue

        res = compute_mvar_lifetime(ctx, v.v, unpack)
        reachable = res.reachable

        ########################################
        # any use that always reads undef?
        for use in v.uses:
            if len(reachable[use].possible_defs) == 0:
                if not use.check_type(ReadMVar).instr.allow_undef:
                    err = use.debug.error("variable is always uninitialized when used here")
                    err.note("in instruction", use)
                    err.note("defined here", v.v.debug)

                @f.replace_instr(use)
                def _():
                    (out_v,) = use.check_type(ReadMVar).outputs_
                    return Move(out_v.type).bind((out_v,), Undef.undef())

                return True

        ########################################
        # remove defs that can never be read
        for d in v.defs:
            (suc,) = graph.successors(d)
            if suc not in reachable:
                if index.instrs[d].parent is None:
                    f.replace_instr(d)(lambda: [])
                    return True

        ########################################
        # try to replace a use with one of the defs
        for use in v.uses:
            # try to replace use with one of the defs

            if index.instrs[use].parent is not None:
                continue

            if reachable[use].possible_undef:
                # here you possibly needs to read a "older" value of the def
                # we cant do the opt in this case
                # ex:
                #
                # while True:
                #     x = f()
                #     if g():
                #         s.write(x)
                #     h(read(s))
                #     return
                #
                # here s may be "x" from the previous iteration of the loop;
                # we can not replace read(s) with x
                #
                # this case is not possible if there is no path external -> [no write] -> read;
                # see logic below
                continue

            def_vals = set(x.inputs_[0] for x in reachable[use].possible_defs)

            if len(def_vals) > 1:
                continue
            (def_val,) = def_vals

            # here we have:
            # the program starts from "external"
            # (1) all paths "external" -> "use" passes through one of {defs}
            # (2) the last element in {defs} this touches must be in {pd}
            # suppose {pd} is all "v := arg" and arg is defined in "argdef"
            # (3) we want to not have: external -[0]-> pd* -[1]-> argdef -[2]-> use. (pd not in [1, 2]) suppose we do:
            # (3.1) all of {pd} depends on argdef, so there is some path external -[3]-> argdef,
            #       where none of {pd} is in [3]
            # (3.2) external -[3]-> argdef -[2]-> use is a path external -> use passing through none of {pd}, contradiction
            # This means that at "use" we can just read the value defined in "argdef".

            @f.replace_instr(use)
            def _():
                (out_v,) = use.check_type(ReadMVar).outputs_
                return Move(v.v.type).bind((out_v,), def_val)

            return True

        ########################################
        # remove defs that writes the same thing as the current value
        for d in v.defs:
            if index.instrs[d].parent is not None:
                continue

            (pred,) = graph.predecessors(d)
            if pred not in reachable:
                continue
            prev_vals = set(x.inputs_[0] for x in reachable[pred].possible_defs)
            (def_val,) = d.check_type(WriteMVar).inputs_
            if prev_vals == {def_val}:
                f.replace_instr(d)(lambda: [])
                return True

    return False


@Transform
def writeback_mvar_use(ctx: TransformCtx):
    f = ctx.frag

    @f.map_instrs
    def _(instr: BoundInstr):
        if i := instr.isinst(ReadMVar):
            (var,) = i.outputs_
            return [
                i,
                WriteMVar(i.instr.s).bind((), var),
            ]
