from .._core import ConstEval
from .._core import Var
from .._instructions import AddI
from .._instructions import Move
from .._instructions import SubI
from .._utils import cast_unchecked
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def opt_consteval(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for instr in index.instrs:
        if instr.instr.consteval_fn is not None and not any(
            isinstance(x, Var) for x in instr.inputs
        ):
            (typ,) = instr.out_types
            (out_var,) = instr.outputs

            if i := instr.isinst(AddI):
                const = ConstEval.addi(*cast_unchecked(i.inputs_))
            elif i := instr.isinst(SubI):
                const = ConstEval.subi(*cast_unchecked(i.inputs_))
            else:
                const = ConstEval.create(
                    typ, instr.instr.consteval_fn, *cast_unchecked(instr.inputs)
                )

            @f.replace_instr(instr)
            def _():
                return Move(typ).bind((out_var,), const)

            return True

    return False
