from stationeers_ic10._instructions import Move

from .._core import BoundInstr
from .._core import Fragment
from .._core import MapInstrsRes
from .._core import ReadMVar
from .._core import WriteMVar
from .._tracing import internal_transform
from .._tracing import mk_var
from .basic import get_basic_index


def _remove_trivial_mvars_once(f: Fragment) -> bool:
    index = get_basic_index(f)

    for v in index.mvars.values():
        if v.private and len(v.defs) == 1:
            rep = mk_var(v.v.type)

            @f.map_instrs
            def _(instr: BoundInstr):
                if instr in v.defs:
                    (arg,) = instr.check_type(WriteMVar).inputs_
                    return Move(v.v.type).bind((rep,), arg)
                if instr in v.uses:
                    (outv,) = instr.check_type(ReadMVar).outputs_
                    return Move(v.v.type).bind((outv,), rep)

            return True

    return False


def _remove_trivial_vars_once(f: Fragment) -> bool:
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

            source.debug.fuse(v.v.debug)

            return True

    return False


@internal_transform
def remove_trivial_vars_mvars(f: Fragment) -> bool:
    changed = False
    while _remove_trivial_mvars_once(f):
        changed = True

    while _remove_trivial_vars_once(f):
        changed = True

    return changed
