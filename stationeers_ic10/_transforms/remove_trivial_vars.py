import networkx as nx

from .._core import BoundInstr
from .._core import Fragment
from .._core import MapInstrsRes
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._instructions import Move
from .._utils import cast_unchecked_val
from .basic import get_basic_index
from .label_provenance import build_instr_flow_graph
from .utils import internal_looping_transform


@internal_looping_transform
def remove_trivial_mvars(f: Fragment) -> bool:
    graph, res = build_instr_flow_graph(f)
    index = res.index

    for v in index.mvars.values():
        if not v.private:
            continue

        if len(v.uses) == 0:

            @f.map_instrs
            def _(instr: BoundInstr) -> MapInstrsRes:
                if instr in v.defs:
                    return []

            return True

        # dict from use -> list of defs that may reach the use
        possible_defs: dict[BoundInstr, list[BoundInstr]] = {u: [] for u in v.uses}

        for d in v.defs:
            graph_ = cast_unchecked_val(graph)(
                nx.restricted_view(  # pyright: ignore[reportUnknownMemberType]
                    graph, set(v.defs) - {d}, []
                ),
            )
            des = nx.descendants(graph_, d)  # pyright: ignore[reportUnknownMemberType]
            for u in v.uses:
                if u in des:
                    possible_defs[u].append(d)

        changed = False

        for u in v.uses:
            pd = possible_defs[u]
            assert len(pd) >= 1
            if len(pd) == 1:

                @f.replace_instr(u)
                def _():
                    (src,) = pd[0].check_type(WriteMVar).inputs_
                    (out_v,) = u.check_type(ReadMVar).outputs_
                    return Move(v.v.type).bind((out_v,), src)

                changed = True

        return changed

    return False


@internal_looping_transform
def remove_trivial_vars(f: Fragment) -> bool:
    index = get_basic_index(f)

    for v in index.vars.values():
        if def_instr := v.def_instr.isinst(Move):
            (source,) = def_instr.inputs_

            @f.map_instrs
            def _(instr: BoundInstr) -> MapInstrsRes:
                if instr == def_instr:
                    return []
                if instr in v.uses:
                    return instr.sub_input_var(v.v, source)

            if isinstance(source, Var):
                source.debug.fuse(v.v.debug)

            return True

    return False
