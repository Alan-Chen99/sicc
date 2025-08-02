from __future__ import annotations

import abc
import functools
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any
from typing import Callable
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
from ._core import Scope
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import can_cast_implicit_many
from ._core import get_type
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import describe_fn
from ._diagnostic import must_use
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._instructions import AddF
from ._instructions import AddI
from ._instructions import BlackBox
from ._instructions import Jump
from ._instructions import PredLE
from ._instructions import PredLT
from ._instructions import UnreachableChecked
from ._tracing import _CUR_SCOPE
from ._tracing import RawSubr
from ._tracing import ensure_label
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import mk_mvar
from ._tracing import trace_if
from ._tracing import trace_to_raw_subr
from ._tree_utils import pytree
from ._utils import Cell
from ._utils import cast_unchecked
from ._utils import empty
from ._utils import empty_t
from ._utils import isinst
from ._utils import late_fn

register_exclusion(__file__)

T_co = TypeVar("T_co", covariant=True, bound=VarT, default=Any)

type UserValue[T: VarT = VarT] = VarRead[T] | T

Bool = UserValue[bool]
Int = UserValue[int]
# TODO: this causes pyright to report
# Type of parameter "x" is unknown (reportUnknownParameterType)
# on function "greater_than"
# when using --threads, sometimes
# type Float = UserValue[float]
type Float = VarRead[float] | float

type ValLabelLike = UserValue[Label] | str


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
    __le__ = late_fn(lambda: less_than_or_eq)

    __gt__ = late_fn(lambda: greater_than)
    __ge__ = late_fn(lambda: greater_than_or_eq)


def _get[T: VarT](v: UserValue[T]) -> Value[T]:
    if isinstance(v, VarT):
        return v
    return v._read()


def _get_label(l: ValLabelLike) -> Value[Label]:
    if isinstance(l, str):
        return ensure_label(l)
    return _get(l)


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

    def __init__(self, x: type[T] | UserValue[T]) -> None:
        if isinstance(x, type):
            self._inner = mk_mvar(x)
            return

        x_val = _get(x)
        self._inner = mk_mvar(get_type(x_val))
        self._inner.write(x_val)

    @staticmethod
    def _from_val[T1: VarT](v: Value[T1]) -> Variable[T1]:
        ans = Variable(get_type(v))
        ans._inner.write(v)
        return ans

    def _read(self) -> Var[T]:
        return self._inner.read()

    @property
    def value(self) -> Variable[T]:
        return Variable(self)

    @value.setter
    def value(self, v: UserValue[T]):
        self._inner.write(_get(v))


class Function[Ts: tuple[Any, ...]]:
    _instrs: Final[Ts]

    def __init__(self, *overloads: Unpack[Ts]) -> None:
        for x in overloads:
            assert isinstance(x, InstrBase)
        self._instrs = cast_unchecked(overloads)

    def __repr__(self) -> str:
        return "Function(" + ", ".join(type(x).__name__ for x in self._instrs) + ")"

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
            if not can_cast_implicit_many(arg_types, instr.in_types):
                continue

            argvals = tuple(_get(x) for x in args)
            with must_use(), track_caller():
                ans = cast_unchecked(instr.call(*argvals))  # pyright: ignore
            if ans is None:
                return None
            assert isinst(ans, Var)

            return Variable._from_val(ans)

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


@dataclass
class State[T = Any]:
    _scope: Scope
    _tree: pytree.PyTreeSpec | None = None
    _vars: list[MVar] | None = None

    def __init__(self, init: T | empty_t = empty):
        self._scope = _CUR_SCOPE.value
        if not isinstance(init, empty_t):
            self.write(init)

    def read(self) -> T:
        assert self._tree is not None
        assert self._vars is not None

        with track_caller():
            vars_ = [x.read() for x in self._vars]
        vars = [Variable._from_val(x) for x in vars_]
        return cast_unchecked(pytree.unflatten(self._tree, vars))

    def write(self, v: T):
        vars, tree = pytree.flatten(cast_unchecked(v))
        vars = [_get(x) for x in vars]

        if self._tree is None:
            assert self._vars is None
            with _CUR_SCOPE.bind(self._scope):
                self._tree = tree
                self._vars = [mk_mvar(get_type(x)) for x in vars]
        else:
            assert self._vars is not None
            assert tree == self._tree
            assert len(vars) == len(self._vars)
        with track_caller():
            for mv, arg in zip(self._vars, vars):
                mv.write(arg)


################################################################################


@dataclass
class TracedSubr[F = Any]:
    subr: RawSubr
    arg_tree: pytree.PyTreeSpec
    ret_tree: pytree.PyTreeSpec

    @property
    def call[**P, R](self: TracedSubr[Callable[P, R]]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            vals = self.arg_tree.flatten_up_to(cast_unchecked((args, kwargs)))
            vals_ = tuple(_get(cast_unchecked(x)) for x in vals)
            out_vars = self.subr.call(*vals_)

            out_vars_ = [Variable._from_val(x) for x in out_vars]
            return cast_unchecked(pytree.unflatten(self.ret_tree, out_vars_))

        return inner


_RETURN_HOOK: Cell[Callable[[Any], None]] = Cell()


def return_(val: Any = None) -> None:
    return _RETURN_HOOK.value(val)


def trace_to_subr[**P, R](
    fn: Callable[P, R | None], *args: P.args, **kwargs: P.kwargs
) -> TracedSubr[Callable[P, R]]:
    arg_vars, arg_tree = pytree.flatten(cast_unchecked((args, kwargs)))
    arg_types = tuple(_get_type(x) for x in arg_vars)

    out_tree: Cell[pytree.PyTreeSpec] = Cell()

    @functools.wraps(fn)
    def inner(*args: Var) -> tuple[Var, ...]:
        ar, kw = arg_tree.unflatten(Variable._from_val(x) for x in args)

        exit = mk_internal_label("trace_to_subr_ret")

        ret_state = State()

        def ret_hook(val: Any) -> None:
            ret_state.write(val)
            jump(exit)

        with _RETURN_HOOK.bind(ret_hook):
            ans = cast_unchecked(fn)(*cast_unchecked(ar), **cast_unchecked(kw))
            if ans is not None:
                with add_debug_info(
                    DebugInfo(describe=f"return val from end of {describe_fn(fn)}")
                ):
                    return_(ans)
            elif ret_state._tree is None:
                return_(None)

            _f.unreachable_checked()

        label(exit)

        assert ret_state._tree is not None
        assert ret_state._vars is not None
        out_tree.value = ret_state._tree
        return tuple(x.read() for x in ret_state._vars)

    ans = trace_to_raw_subr(arg_types, inner)

    return TracedSubr(ans, arg_tree, out_tree.value)


class Subr[F]:
    fn: Final[F]

    def __init__(self, fn: F) -> None:
        self.fn = fn
        self._subr: TracedSubr[F] | None = None

    @property
    def __call__[**P, R](self: Subr[Callable[P, R]]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            if self._subr is None:
                self._subr = trace_to_subr(self.fn, *args, **kwargs)
                # FIXME?
                # emit_frag(self._subr.subr.frag)
            return self._subr.call(*args, **kwargs)

        return inner


def subr():
    def inner[F](fn: F) -> Subr[F]:
        return Subr(fn)

    return inner


################################################################################

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


def jump(label: ValLabelLike) -> None:
    return Jump().call(_get_label(label))


def branch(cond: Bool, on_true: ValLabelLike, on_false: ValLabelLike) -> None:
    return _f.branch(_get(cond), _get_label(on_true), _get_label(on_false))


def if_(cond: Bool) -> AbstractContextManager[None]:
    return trace_if(_get(cond))


################################################################################
