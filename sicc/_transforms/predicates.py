import math

from .._instructions import PredEq
from .._instructions import PredNAN
from .._instructions import PredNEq
from .._instructions import PredNotNAN
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def handle_nan_eq(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for instr in index.instrs:
        if i := instr.isinst((PredEq, PredNEq)):
            new = PredNotNAN() if instr.isinst(PredNEq) else PredNAN()
            x, y = i.inputs_
            if isinstance(x, float) and math.isnan(x):
                arg = y
            elif isinstance(y, float) and math.isnan(y):
                arg = x
            else:
                continue

            @f.replace_instr(i)
            def _():
                return new.bind(i.outputs_, arg)

            return True

    return False
