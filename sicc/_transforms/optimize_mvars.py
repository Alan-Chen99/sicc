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
from .._core import Var
from .._core import WriteMVar
from .._instructions import Move
from .._utils import cast_unchecked_val
from .._utils import isinst
from ..config import verbose
from .basic import get_index
from .control_flow import CfgNode
from .control_flow import build_control_flow_graph
from .control_flow import external
from .utils import LoopingTransform
from .utils import Transform
from .utils import TransformCtx


@dataclass
class LifetimeResPerInstr:
    # only constructed if cur is not a [def and not use]

    # the mvar value BEFORE cur may be used latter at ...
    # note: this means:
    # includes cur, if cur is a use, or a [def and use]
    # if cur is WriteMvar, this would be empty by definition;
    # note that cur would then not be in reachable and we dont construct the object in this case
    possible_uses: list[BoundInstr]

    # the value of the mvar BEFORE cur may come from ...
    # if possible uses is empty, this is empty too, even if it may have a value
    # a def -> external -> cur counts
    possible_defs: list[BoundInstr]

    # is there a path external -> cur touching no defs?
    external_path: bool = False


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
    """
    note: previously operations other than WriteMVar and ReadMVar is not supported
    this is no longer the case and now this function only checks if v is private
    this function may get removed in future
    """
    index = get_index.call_cached(ctx, unpack)
    v = index.mvars[v_]

    if not v.private:
        return False

    # # not sufficiently expanded
    # if not all(x.isinst(WriteMVar) for x in v.defs):
    #     return False
    # if not all(x.isinst(ReadMVar) for x in v.uses):
    #     return False

    return True


def compute_mvar_lifetime(
    ctx: TransformCtx, v_: MVar, unpack: UnpackPolicy = NeverUnpack()
) -> MvarLifetimeRes:
    graph = build_control_flow_graph.call_cached(ctx, out_unpack=unpack)
    index = get_index.call_cached(ctx, unpack)

    assert support_mvar_analysis(ctx, v_, unpack=unpack)
    v = index.mvars[v_]

    ########################################
    # get the subgraph where a value set to v
    # may get used
    # "ancestors" call wraps around "external"

    hide_defs = cast_unchecked_val(graph)(
        nx.restricted_view(graph, v.defs - v.uses, []),  # pyright: ignore[reportUnknownMemberType]
    )
    res: dict[CfgNode, LifetimeResPerInstr] = {}

    def get(x: CfgNode) -> LifetimeResPerInstr:
        if ans := res.get(x):
            return ans
        ans = LifetimeResPerInstr([], [])
        res[x] = ans
        return ans

    for use in v.uses:
        get(use).possible_uses.append(use)
        for anc in nx.ancestors(hide_defs, use):  # pyright: ignore[reportUnknownMemberType]
            get(anc).possible_uses.append(use)

    ########################################

    # does not include any defs
    # include external, if it can reach any use without reaching a def first
    reachable = set(res)

    for d in v.defs:
        subgraph = graph.subgraph(reachable | {d})
        des = nx.descendants(subgraph, d)  # pyright: ignore[reportUnknownMemberType]1
        for x in des:
            res[x].possible_defs.append(d)

    subgraph = graph.subgraph(reachable | {external})
    des = nx.descendants(subgraph, external)  # pyright: ignore[reportUnknownMemberType]1
    for x in des:
        res[x].external_path = True

    if external in reachable:
        res[external].external_path = True
    else:
        assert len(des) == 0

    ########################################

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
            if not (use := use.isinst(ReadMVar)):
                continue

            if len(reachable[use].possible_defs) == 0:
                err = use.debug.error("variable is always uninitialized when used here")
                if verbose.value >= 1:
                    err.note("in instruction", use)
                err.note("defined here", v.v.debug)

                @f.replace_instr(use)
                def _():
                    (out_v,) = use.outputs_
                    return Move(out_v.type).bind((out_v,), Undef(out_v.type))

                return True

        ########################################
        # remove defs that can never be read
        for d in v.defs:
            if not (d := d.isinst(WriteMVar)):
                continue

            (suc,) = graph.successors(d)
            if suc not in reachable:
                f.replace_instr(d)(lambda: [])
                return True

        ########################################
        # replace a use with one of the defs
        for use in v.uses:
            if not (use := use.isinst(ReadMVar)):
                continue

            possible_defs = reachable[use].possible_defs

            if any(not d.isinst(WriteMVar) for d in possible_defs):
                continue

            def_instrs = [x.check_type(WriteMVar) for x in possible_defs]
            def_vals = set(x.inputs_[0] for x in def_instrs)

            if len(def_vals) != 1:
                continue
            (def_val,) = def_vals

            # at "use", the value comes from one of the defs: "v := arg"
            # we want to know whether v still have the same value as "arg" at "use"

            # by definition v is not changed, so we want to know whether "arg" gets changed
            # if it does, this happens as:
            # pd -> argdef -> use
            # where this path does not pass through any defs of v

            # concrete examples of problematic case:
            #
            # while True:
            #     x = f() # argdef
            #     if g():
            #         s.write(x) # pd
            #     h(read(s)) # use

            if not isinst(def_val, Var):
                # constant; ok for opt
                pass
            else:
                # abort if exist a path pd -> argdef -> use not passing through any of {pd}

                argdef = index.vars[def_val].def_instr
                subgraph = graph.subgraph(graph.nodes - set(v.defs))

                # result satisfies ssa invariant?
                if nx.has_path(graph.subgraph(graph.nodes - {argdef}), external, use):
                    # result is potentially sound, but not valid ssa
                    continue

                if nx.has_path(subgraph, argdef, use):
                    # has a argdef -> use
                    # we abort here atm
                    # TODO: potentially also check pd -> argdef and allow the opt if no path
                    continue

            @f.replace_instr(use)
            def _():
                (out_v,) = use.outputs_
                ans = Move(v.v.type).bind((out_v,), def_val)
                for pd in def_instrs:
                    ans.debug.fuse_must_use(pd.debug)
                return ans

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
