from typing import cast

import networkx as nx
from rich import print as print  # autoflake: skip

from .._core import BoundInstr
from .._instructions import Move
from .basic import get_index
from .control_flow import CfgNode
from .control_flow import External
from .control_flow import build_control_flow_graph
from .control_flow import external
from .utils import LoopingTransform
from .utils import TransformCtx


def _can_sub_elim(ctx: TransformCtx, instr: BoundInstr, parent: BoundInstr) -> bool:
    if not (parent.instr == instr.instr and parent.inputs == instr.inputs):
        return False
    if instr.is_pure():
        return True
    assert instr.is_side_effect_free()

    index = get_index.call_cached(ctx)
    cfg = build_control_flow_graph.call_cached(ctx)

    reads = instr.reads()

    between = nx.ancestors(  # pyright: ignore[reportUnknownMemberType]
        cfg.subgraph(cfg.nodes - {parent}), instr
    )
    for x in between:
        assert not isinstance(x, External)
        for eff in x.writes():
            for r in reads:
                if index.effect_conflicts.has_edge(eff, r):
                    return False

    return True


@LoopingTransform
def common_sub_elim(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)
    cfg = build_control_flow_graph.call_cached(ctx)

    imm_doms = cast(
        dict[CfgNode, CfgNode],
        nx.immediate_dominators(cfg, external),  # pyright: ignore[reportUnknownMemberType]
    )

    for instr in index.instrs:
        if instr.isinst(Move):
            continue
        if instr.jumps_to() or not instr.continues:
            continue
        if not instr.is_side_effect_free():
            continue

        if instr not in imm_doms:
            # not reachable
            continue
        p = imm_doms[instr]
        while not isinstance(p, External):
            if _can_sub_elim(ctx, instr, p):

                @f.replace_instr(instr)
                def _():
                    assert len(instr.outputs) == len(p.outputs)
                    for instr_out, p_out in zip(instr.outputs, p.outputs):
                        yield Move(instr_out.type).bind((instr_out,), p_out)

                p.debug.must_use_ctx += instr.debug.must_use_ctx
                return True

            p = imm_doms[p]

    return False
