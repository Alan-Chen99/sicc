import math

from .._core import Var
from .._core import VirtualConst
from .._instructions import Move
from .._instructions import PredEq
from .._instructions import PredicateBase
from .._instructions import PredLE
from .._instructions import PredLT
from .._instructions import PredNAN
from .._instructions import PredNEq
from .._instructions import PredNotNAN
from .._utils import cast_unchecked
from .._utils import crc32
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


def _literal_value(x: object) -> float | None:
    if isinstance(x, str):
        # strings are compared at runtime by their HASH
        return crc32(x)
    if isinstance(x, (int, float)):
        return x
    return None


def _const_pred_value(pred: PredicateBase, args: list[float]) -> bool | None:
    if isinstance(pred, PredLT):
        return args[0] < args[1]
    if isinstance(pred, PredLE):
        return args[0] <= args[1]
    if isinstance(pred, (PredEq, PredNEq)):
        x, y = args
        # `== nan` is a nan check; same rule as handle_nan_eq
        if math.isnan(x) or math.isnan(y):
            eq = math.isnan(x) and math.isnan(y)
        else:
            eq = x == y
        return eq if isinstance(pred, PredEq) else not eq
    if isinstance(pred, PredNAN):
        return math.isnan(args[0])
    if isinstance(pred, PredNotNAN):
        return not math.isnan(args[0])
    return None


def _eval_const_pred(pred: PredicateBase, inputs: tuple[object, ...]) -> bool | None:
    args = [_literal_value(x) for x in inputs]
    nums = [x for x in args if x is not None]
    if len(nums) == len(args):
        return _const_pred_value(pred, nums)

    if isinstance(pred, (PredEq, PredNEq)):
        x, y = inputs
        # e.g. the same Label is the same line. distinct Labels are undecided: they can still
        # resolve to the same line. a VirtualConst is only known when emitting.
        if x == y and not isinstance(x, (Var, VirtualConst)):
            return isinstance(pred, PredEq)

    return None


@LoopingTransform
def fold_const_predicates(ctx: TransformCtx) -> bool:
    """replace a predicate on constants with its value, where that is known before emitting"""
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for instr in index.instrs:
        if not (i := instr.isinst(PredicateBase)):
            continue
        val = _eval_const_pred(i.instr, i.inputs)
        if val is None:
            continue

        @f.replace_instr(i)
        def _():
            (out_var,) = i.outputs
            return Move(bool).bind((cast_unchecked(out_var),), val)

        return True

    return False
