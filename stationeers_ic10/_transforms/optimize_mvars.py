from dataclasses import dataclass

import networkx as nx
from ordered_set import OrderedSet

from .._core import MVar
from .._core import NeverUnpack
from .._core import ReadMVar
from .._core import UnpackPolicy
from .._core import Value
from .._core import WriteMVar
from .._instructions import Move
from .._utils import cast_unchecked_val
from .basic import get_index
from .control_flow import CfgNode
from .control_flow import External
from .control_flow import build_control_flow_graph
from .control_flow import external
from .utils import LoopingTransform
from .utils import TransformCtx


@dataclass
class MvarLifetimeRes:
    v: MVar
    #: set of instr where the value of v at the instr may get used latter
    #: does not include any def or uses of v
    reachable: OrderedSet[CfgNode]


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

    ########################################
    # get the subgraph where a value set to v
    # may get used
    # "ancestors" call wraps around "external"

    hide_defs = cast_unchecked_val(graph)(
        nx.restricted_view(graph, v.defs, []),  # pyright: ignore[reportUnknownMemberType]
    )
    reachable: set[CfgNode] = set(v.uses)
    # NOTE: this can be faster
    for use in v.uses:
        reachable |= nx.ancestors(hide_defs, use)  # pyright: ignore[reportUnknownMemberType]

    ########################################

    # sort since set not deterministic
    return MvarLifetimeRes(v.v, reachable=OrderedSet(sorted(reachable)))


@LoopingTransform
def elim_mvars_read_writes(ctx: TransformCtx, unpack: UnpackPolicy = NeverUnpack()) -> bool:
    """
    # (1) optimize mvar reads that can only come from a single mvar write
    # (2) remove mvar write that can never b read
    """
    f = ctx.frag
    graph = build_control_flow_graph.call_cached(ctx, out_unpack=unpack)
    index = get_index.call_cached(ctx, unpack)

    for v in index.mvars.values():
        if not support_mvar_analysis(ctx, v.v):
            continue

        reachable = compute_mvar_lifetime(ctx, v.v).reachable

        # remove defs that is not reachable
        for d in v.defs:
            (suc,) = graph.successors(d)
            if suc not in reachable:
                f.replace_instr(d)(lambda: [])
                return True

        subgraph = graph.subgraph(reachable | set(v.defs))

        possible_defs: dict[CfgNode, set[Value | External]] = {u: set() for u in subgraph.nodes}

        # here values may "wrap around" if it goes def -> jump out -> jump back -> use
        # we make a special def "external" representing any wrapped around value
        # and dont do the optimization in any place this special def effects
        # this is needed for correctness in logic below
        defs = list(v.defs)
        if external in reachable:
            defs += [external]

        for d in defs:
            if isinstance(d, External):
                def_val = external
            else:
                (def_val,) = d.check_type(WriteMVar).inputs_

            graph_ = cast_unchecked_val(subgraph)(
                nx.restricted_view(  # pyright: ignore[reportUnknownMemberType]
                    subgraph, set(defs) - {d}, []
                ),
            )
            des = nx.descendants(graph_, d)  # pyright: ignore[reportUnknownMemberType]1
            for x in des:
                possible_defs[x].add(def_val)

        for use in v.uses:
            pd = possible_defs[use]
            assert len(pd) >= 1

            # TODO: if External is in pd, mvar may be undefined, maybe emit warning?

            if len(pd) > 1:
                continue
            (def_val,) = list(pd)

            if isinstance(def_val, External):
                # note: we cant do the opt in this case
                # ex:
                #
                # start:
                # x = f()
                # if g():
                #    s.write(x)
                # h(read(s))
                # return
                #
                # here s may be undefined, and we should not replace read(s) with x
                continue

            # here we have:
            # the program starts from "external"
            # (1) all paths "external" -> "use" passes through one of {defs} - {external}
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

        for d in v.defs:

            # check if d writes the same thing as a previous write
            (pred,) = graph.predecessors(d)
            if pred not in possible_defs:
                continue
            prev_vals = list(possible_defs[pred])
            (def_val,) = d.check_type(WriteMVar).inputs_
            if prev_vals == [def_val]:
                f.replace_instr(d)(lambda: [])
                return True

    return False
