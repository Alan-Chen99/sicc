from .._core import InstrBase
from .._core import VarT
from .._instructions import AddF
from .._instructions import AddI
from .._instructions import DivF
from .._instructions import Move
from .._instructions import MulF
from .._instructions import MulI
from .._instructions import SubF
from .._instructions import SubI
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx

_right_identity: dict[type[InstrBase], VarT] = {
    AddI: 0,
    AddF: 0,
    SubI: 0,
    SubF: 0,
    MulI: 1,
    MulF: 1,
    DivF: 1,
}


_left_identity: dict[type[InstrBase], VarT] = {
    AddI: 0,
    AddF: 0,
    MulI: 1,
    MulF: 1,
}


@LoopingTransform
def opt_identity_arith(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for instr in index.instrs:
        if (id_val := _right_identity.get(type(instr.instr))) is not None:
            a, b = instr.inputs
            if b == id_val:

                @f.replace_instr(instr)
                def _():
                    (output,) = instr.outputs
                    (out_type,) = instr.out_types
                    return Move(out_type).bind((output,), a)

                return True

        if (id_val := _left_identity.get(type(instr.instr))) is not None:
            a, b = instr.inputs
            if a == id_val:

                @f.replace_instr(instr)
                def _():
                    (output,) = instr.outputs
                    (out_type,) = instr.out_types
                    return Move(out_type).bind((output,), b)

                return True

    return False
