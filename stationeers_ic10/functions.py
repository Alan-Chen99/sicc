from __future__ import annotations

from typing import overload

from ._core import Bool, Float, Int, ValLabel, Value, Var, VarT, can_cast_val, get_type
from ._instructions import AddF, AddI, BlackBox, Branch, Jump, PredVar, UnreachableChecked

jump = Jump().call
unreachable_checked = UnreachableChecked().call


@overload
def add(x: Int, y: Int) -> Var[int]: ...
@overload
def add(x: Float, y: Float) -> Var[float]: ...
def add(x: Float, y: Float) -> Var[float]:
    if can_cast_val(x, int) and can_cast_val(y, int):
        return AddI().call(x, y)
    return AddF().call(x, y)


def black_box[T: VarT](v: Value[T]) -> Var[T]:
    return BlackBox(get_type(v)).call(v)


def branch(cond: Bool, on_true: ValLabel, on_false: ValLabel) -> None:
    return Branch(PredVar()).call(on_true, on_false, cond)
