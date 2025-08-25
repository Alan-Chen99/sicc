from collections import defaultdict
from typing import cast

import networkx as nx
from rich import print as print  # autoflake: skip

from .._core import BoundInstr
from .._core import Const
from .._core import ConstEval
from .._core import UnpackPolicy
from .._core import Var
from .._instructions import AddI
from .._instructions import MoveBase
from .._instructions import Pop
from .._instructions import Push
from .._instructions import SubI
from .._utils import cast_unchecked
from .basic import get_index
from .basic import remove_unused_side_effect_free
from .control_flow import CfgNode
from .control_flow import build_control_flow_graph
from .control_flow import external
from .utils import LoopingTransform
from .utils import TransformCtx


class UnpackPushPop(UnpackPolicy):
    def should_unpack(self, instr: BoundInstr) -> bool:
        return bool(instr.isinst((Push, Pop)))


def _as_offset_arith(instr: BoundInstr) -> tuple[Var[int], Var[int], Const[int]] | None:
    if i := instr.isinst(AddI):
        x, y = i.inputs_
        (o,) = i.outputs_
        if isinstance(x, Var) and not isinstance(y, Var):
            return o, x, y
        if isinstance(y, Var) and not isinstance(x, Var):
            return o, y, x

    if i := instr.isinst(SubI):
        x, y = i.inputs_
        (o,) = i.outputs_
        if isinstance(x, Var) and not isinstance(y, Var):
            return o, x, ConstEval.subi(0, y)

    # perhaps should be a subset of MoveBase instead?
    if (i := instr.isinst(MoveBase)) and i.in_types == (int,) and i.out_types == (int,):
        (ipt,) = i.inputs_
        (opt,) = i.outputs_
        if isinstance(ipt, Var):
            return opt, cast_unchecked(ipt), 0

    return None


@LoopingTransform
def opt_offset_arith(ctx: TransformCtx) -> bool:
    f = ctx.frag
    if remove_unused_side_effect_free(f):
        return True

    index = get_index.call_cached(ctx, UnpackPushPop())
    cfg = build_control_flow_graph.call_cached(ctx, out_unpack=UnpackPushPop())

    imm_doms = cast(
        dict[CfgNode, CfgNode],
        nx.immediate_dominators(cfg, external),  # pyright: ignore[reportUnknownMemberType]
    )

    children: defaultdict[CfgNode, list[CfgNode]] = defaultdict(list)

    for x, y in imm_doms.items():
        if x != external:
            children[y].append(x)
    # ensure deterministic even if nx is not
    for l in children.values():
        l.sort()

    offsets_map: dict[Var[int], tuple[Var[int], Const[int]]] = {}
    avail_offsets: defaultdict[Var[int], list[tuple[Var[int], Const[int]]]] = defaultdict(list)

    def is_removable(instr: BoundInstr) -> bool:
        return bool(instr.isinst((AddI, SubI)) and index.instrs[instr].parent is None)

    def walk(cur: CfgNode) -> bool:
        if isinstance(cur, BoundInstr) and (specs := _as_offset_arith(cur)) and is_removable(cur):
            o, base, offset = specs
            root, base_offset = offsets_map.get(base, (base, 0))
            tot_offset = ConstEval.addi(base_offset, offset)

            # typical sequence:
            # v:= root + 1
            # f(v)
            # x:= v + 2
            # o:= x + 3

            # we want to make the last one
            # o:= v + 5

            for v, vo in reversed(avail_offsets[root]):
                # look for the "best" possible v to base on

                if len(index.vars[v].uses) > 1 or not is_removable(index.vars[v].def_instr):
                    # good choice
                    pass
                else:
                    # keep searching
                    continue

                if base == v:
                    break

                # make cur based on v instead
                @f.replace_instr(cur)
                def _():
                    return AddI().bind((o,), v, ConstEval.subi(tot_offset, vo))

                return True

        if isinstance(cur, BoundInstr):
            # possible to replace input with the result %sp of a push pop chain?

            for arg in cur.inputs:
                if not isinstance(arg, Var):
                    continue

                if specs := offsets_map.get(cast_unchecked(arg)):
                    # have a root
                    root, offset = specs
                elif arg in avail_offsets:
                    # is a root
                    root, offset = cast(Var[int], arg), 0
                else:
                    continue

                for v, vo in reversed(avail_offsets[root]):
                    if arg == v:
                        break

                    if vo == offset:

                        @f.replace_instr(cur)
                        def _():
                            return cur.sub_val(arg, v, inputs=True)

                        return True

        if isinstance(cur, BoundInstr) and (specs := _as_offset_arith(cur)):
            o, base, offset = specs
            root, base_offset = offsets_map.get(base, (base, 0))
            tot_offset = ConstEval.addi(base_offset, offset)

            offsets_map[o] = root, tot_offset
            if root in avail_offsets and avail_offsets[root]:
                avail_offsets[root].append((o, tot_offset))
            else:
                avail_offsets[root] = [(root, 0), (o, tot_offset)]
            for child in children[cur]:
                if walk(child):
                    return True
            avail_offsets[root].pop()

        else:
            for child in children[cur]:
                if walk(child):
                    return True

        return False

    if walk(external):
        return True

    return False
