from __future__ import annotations

import abc
from typing import Any
from typing import Final
from typing import Protocol
from typing import TypeVar
from typing import Unpack
from typing import overload

from . import _functions as _f
from ._core import InstrBase
from ._core import InstrTypedWithArgs_api
from ._core import Label
from ._core import MVar
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import can_cast_implicit_many
from ._instructions import AddF
from ._instructions import AddI
from ._instructions import BlackBox
from ._instructions import Jump
from ._instructions import PredLE
from ._instructions import PredLT
from ._instructions import UnreachableChecked
from ._tracing import mk_mvar
from ._tracing import trace_if
from ._utils import cast_unchecked
from ._utils import isinst
from ._utils import late_fn

T_co = TypeVar("T_co", covariant=True, bound=VarT, default=Any)

type UserValue[T: VarT = VarT] = VarRead[T] | T

Bool = UserValue[bool]
Int = UserValue[int]
Float = UserValue[float]
ValLabel = UserValue[Label]


class VarRead[T: VarT](abc.ABC):
    """
    covariant readonly base
    """

    @abc.abstractmethod
    def _read(self) -> Var[T]: ...

    def __check_co(self) -> VarRead[VarT]:  # pyright: ignore[reportUnusedFunction]
        return self

    __add__ = late_fn(lambda: add)
    __radd__ = late_fn(lambda: add)

    __lt__ = late_fn(lambda: less_than)
    __gt__ = late_fn(lambda: greater_than)

    __le__ = late_fn(lambda: less_than_or_eq)
    __ge__ = late_fn(lambda: greater_than_or_eq)


def _get[T: VarT](v: UserValue[T]) -> Value[T]:
    if isinstance(v, VarT):
        return v
    return v._read()


def _get_type[T: VarT](v: UserValue[T]) -> type[T]:
    if isinstance(v, VarT):
        return type(v)
    if isinstance(v, Variable):
        return v._inner.type
    assert False


class Variable[T: VarT](VarRead[T]):
    """
    a MVar exposed to the user.

    This class is needed because Var/MVar have equality and compareison by id
    (needed in compiler internals); Variable equality will instead stage a equality operation.
    """

    _inner: MVar[T]

    def __init__(self, x: type[T] | T) -> None:
        if isinstance(x, type):
            typ = x
        else:
            typ = type(x)

        self._inner = mk_mvar(typ)

    def _read(self) -> Var[T]:
        return self._inner.read()

    @property
    def value(self):
        return self._read()

    @value.setter
    def value(self, v: UserValue[T]):
        self._inner.write(_get(v))


class Function[Ts: tuple[Any, ...]]:
    _instrs: Final[Ts]

    def __init__(self, *overloads: Unpack[Ts]) -> None:
        for x in overloads:
            assert isinstance(x, InstrBase)
        self._instrs = cast_unchecked(overloads)

    def __get__[V](self, obj: V, objtype: Any) -> _BoundFunction[V, Ts]:
        """
        typing helper to get type checker accept Function as a method
        for methods to actually work one must use for ex late_fn
        """
        assert False

    @overload
    def __call__[*I, O](
        self: Function[tuple[InstrTypedWithArgs_api[tuple[*I], O], *tuple[Any, ...]]], *args: *I
    ) -> O: ...
    @overload
    def __call__[*I, O](
        self: Function[tuple[Any, InstrTypedWithArgs_api[tuple[*I], O], *tuple[Any, ...]]],
        *args: *I,
    ) -> O: ...

    def __call__(self, *args: Any) -> Any:
        arg_types = tuple(_get_type(x) for x in args)

        for instr in self._instrs:
            assert isinstance(instr, InstrBase)
            if can_cast_implicit_many(arg_types, instr.in_types):
                ans = cast_unchecked(
                    instr.call(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                        *(_get(x) for x in args)
                    )
                )
                if ans is None:
                    return None
                assert isinst(ans, Var)

                ans_var = Variable(ans.type)
                ans_var._inner.write(ans)
                return ans_var

        raise TypeError()


class _BoundFunction[V, Ts: tuple[Any, ...]](Protocol):
    @overload
    def __call__[V1, *I, O](
        self: _BoundFunction[V1, tuple[InstrTypedWithArgs_api[tuple[V1, *I], O], *tuple[Any, ...]]],
        *args: *I,
    ) -> O: ...
    @overload
    def __call__[V1, *I, O](
        self: _BoundFunction[
            V1, tuple[Any, InstrTypedWithArgs_api[tuple[V1, *I], O], *tuple[Any, ...]]
        ],
        *args: *I,
    ) -> O: ...


################################################################################

jump = Function(Jump())

add = Function(AddI(), AddF())

unreachable_checked = Function(UnreachableChecked())

less_than = Function(PredLT())
less_than_or_eq = Function(PredLE())


def greater_than(x: Float, y: Float) -> Variable[bool]:
    return less_than(y, x)


def greater_than_or_eq(x: Float, y: Float) -> Variable[bool]:
    return less_than_or_eq(y, x)


def black_box[T: VarT](v: UserValue[T]) -> Variable[T]:
    return Function(BlackBox(_get_type(v)))(v)


################################################################################


def branch(cond: Bool, on_true: ValLabel, on_false: ValLabel) -> None:
    return _f.branch(_get(cond), _get(on_true), _get(on_false))


def if_(cond: Bool):
    return trace_if(_get(cond))


################################################################################
