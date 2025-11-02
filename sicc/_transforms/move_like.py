from rich import print as print  # autoflake: skip

from .._core import Var
from .._instructions import Move
from .._instructions import MoveBase
from .._instructions import Transmute
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def opt_move_like(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for instr in index.instrs:
        if (i := instr.isinst(MoveBase)) and not instr.isinst((Move, Transmute)):
            (arg,) = i.inputs_
            if not isinstance(arg, Var):

                @f.replace_instr(i)
                def _():
                    return Move(i.out_types[0]).bind(i.outputs_, arg)

                return True

    return False
