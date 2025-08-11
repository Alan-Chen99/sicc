from __future__ import annotations

import abc
import functools
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Final
from typing import Never
from typing import Protocol
from typing import Self
from typing import Sequence
from typing import TypedDict
from typing import TypeVar
from typing import Unpack
from typing import overload
from typing import override

from ordered_set import OrderedSet
from rich.text import Text

from . import _functions as _f
from ._core import AnyType
from ._core import BoundInstr
from ._core import Comment
from ._core import InstrBase
from ._core import InstrTypedWithArgs_api
from ._core import Label
from ._core import MVar
from ._core import Scope
from ._core import TypeList
from ._core import Undef
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import can_cast_implicit_many
from ._core import can_cast_implicit_many_or_err
from ._core import get_type
from ._core import get_types
from ._core import promote_types
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import describe_fn
from ._diagnostic import must_use
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._instructions import AddF
from ._instructions import AddI
from ._instructions import AndB
from ._instructions import AndI
from ._instructions import AsmBlock
from ._instructions import AsmBlockInner
from ._instructions import BlackBox
from ._instructions import DivF
from ._instructions import EffectExternal
from ._instructions import Jump
from ._instructions import Move
from ._instructions import MulF
from ._instructions import MulI
from ._instructions import Not
from ._instructions import OrB
from ._instructions import OrI
from ._instructions import PredEq
from ._instructions import PredLE
from ._instructions import PredLT
from ._instructions import PredNEq
from ._instructions import RawInstr
from ._instructions import Select
from ._instructions import SubF
from ._instructions import SubI
from ._instructions import UnreachableChecked
from ._tracing import _CUR_SCOPE
from ._tracing import RawSubr
from ._tracing import ensure_label
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import mk_mvar
from ._tracing import trace_if
from ._tracing import trace_to_raw_subr
from ._tracing import trace_while
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
Str = UserValue[str]

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

    @abc.abstractmethod
    def _get_type(self) -> type[T]: ...

    def __check_co(self) -> VarRead[VarT]:  # pyright: ignore[reportUnusedFunction]
        return self

    def __bool__(self) -> Never:
        err = TypeError("not possible to convert a runtime value to a compile-time bool.")
        err.add_note("use tilde operator '~' to invert a boolean")
        raise err

    __add__ = late_fn(lambda: add)
    __radd__ = late_fn(lambda: add)

    __sub__ = late_fn(lambda: sub)
    __rsub__ = late_fn(lambda: sub._rev_same_sig())

    __mul__ = late_fn(lambda: mul)
    __rmul__ = late_fn(lambda: mul)

    __truediv__ = late_fn(lambda: div)
    __rtruediv__ = late_fn(lambda: div._rev_same_sig())

    __invert__ = late_fn(lambda: bool_not)

    __and__ = late_fn(lambda: and_)
    __or__ = late_fn(lambda: or_)

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Any
    ) -> VarRead[bool]:
        return equal(self, other)

    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Any
    ) -> VarRead[bool]:
        return not_equal(self, other)

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


def mk_label(l: str | None = None) -> Label:
    """make a label to be emitted later"""
    return ensure_label(l)


def _get_type[T: VarT](v: UserValue[T]) -> type[T]:
    if isinstance(v, VarT):
        return type(v)
    if not isinstance(v, VarRead):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"unsupported type: {type(v)} (value: {v})"
        )
    return v._get_type()


class VariableOpts(TypedDict, total=False):
    _read_only: bool


class Variable[T: VarT](VarRead[T]):
    """
    a MVar exposed to the user.

    This class is needed because Var/MVar have equality and compareison by id
    (needed in compiler internals); Variable equality will instead stage a equality operation.
    """

    _inner: MVar[T]
    _read_only: bool

    @overload
    def __init__(self, typ: type[T], /, **kwargs: Unpack[VariableOpts]) -> None: ...
    @overload
    def __init__(self, init: UserValue[T], /, **kwargs: Unpack[VariableOpts]) -> None: ...
    @overload
    def __init__(
        self, typ: type[T], init: UserValue[T], /, **kwargs: Unpack[VariableOpts]
    ) -> None: ...
    def __init__(
        self,
        init_or_typ: type[T] | UserValue[T],
        init: UserValue[T] | None = None,
        /,
        **kwargs: Unpack[VariableOpts],
    ) -> None:
        read_only = kwargs.get("_read_only", False)

        self._read_only = read_only

        if init is not None:
            assert isinstance(init_or_typ, type)
            self._inner = mk_mvar(init_or_typ)
            self._inner.write(_get(init))
            return

        if isinstance(init_or_typ, type):
            self._inner = mk_mvar(init_or_typ)
            return

        x_val = _get(init_or_typ)
        self._inner = mk_mvar(get_type(x_val))
        self._inner.write(x_val)

    def __repr__(self) -> str:
        return repr(self._inner)

    @staticmethod
    def _from_val_ro[T1: VarT](v: Value[T1]) -> Variable[T1]:
        ans = Variable(get_type(v), _read_only=True)
        ans._inner.write(v)
        return ans

    @override
    def _read(self) -> Var[T]:
        return self._inner.read()

    @override
    def _get_type(self) -> type[T]:
        return self._inner.type

    @property
    def value(self) -> VarRead[T]:
        return Variable(self, _read_only=True)

    @value.setter
    def value(self, v: UserValue[T]):
        if self._read_only:
            raise TypeError(f"writing read-only variable {self}")
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

            return Variable._from_val_ro(ans)

        can_cast_implicit_many_or_err(arg_types, self._instrs[-1].in_types)
        assert False

    call = __call__

    def _rev_same_sig(self) -> Self:
        def inner(a1: Any, a2: Any) -> Any:
            return self.call(a2, a1)

        return cast_unchecked(inner)


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
        vars = [Variable._from_val_ro(x) for x in vars_]
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

            out_vars_ = [Variable._from_val_ro(x) for x in out_vars]
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
        ar, kw = arg_tree.unflatten(Variable._from_val_ro(x) for x in args)

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
            return self._subr.call(*args, **kwargs)

        return inner


def subr():
    def inner[F](fn: F) -> Subr[F]:
        return Subr(fn)

    return inner


################################################################################


def undef() -> VarRead[Never]:
    """return the undef constant"""
    return Variable(Undef.undef(), _read_only=True)


################################################################################

add = Function(AddI(), AddF())
sub = Function(SubI(), SubF())
mul = Function(MulI(), MulF())
div = Function(DivF())

or_ = Function(OrB(), OrI())
and_ = Function(AndB(), AndI())

unreachable_checked = Function(UnreachableChecked())

bool_not = Function(Not())

less_than = Function(PredLT())
less_than_or_eq = Function(PredLE())


def greater_than(x: Float, y: Float) -> VarRead[bool]:
    return less_than(y, x)


def greater_than_or_eq(x: Float, y: Float) -> VarRead[bool]:
    return less_than_or_eq(y, x)


def equal[T: VarT](x: UserValue[T], y: UserValue[T]) -> VarRead[bool]:
    return Function(PredEq(promote_types(_get_type(x), _get_type(y)))).call(x, y)


def not_equal[T: VarT](x: UserValue[T], y: UserValue[T]) -> VarRead[bool]:
    return Function(PredNEq(promote_types(_get_type(x), _get_type(y)))).call(x, y)


def black_box[T: VarT](v: UserValue[T]) -> VarRead[T]:
    return Function(BlackBox(_get_type(v))).call(v)


def comment(x: str, *args: UserValue) -> None:
    args_ = [_get(v) for v in args]
    # with track_caller():
    return Comment(Text(x, "ic10.comment"), *get_types(*args_)).call(*args_)


################################################################################


def jump(label: ValLabelLike) -> None:
    return Jump().call(_get_label(label))


def branch(cond: Bool, on_true: ValLabelLike, on_false: ValLabelLike) -> None:
    return _f.branch(_get(cond), _get_label(on_true), _get_label(on_false))


def if_(cond: Bool) -> AbstractContextManager[None]:
    return trace_if(_get(cond))


def while_(cond_fn: Callable[[], Bool]) -> AbstractContextManager[None]:
    def inner():
        return _get(cond_fn())

    return trace_while(inner)


def loop() -> AbstractContextManager[None]:
    return while_(lambda: True)


################################################################################


@overload
def select[T: VarT](pred: Bool, on_true: UserValue[T], on_false: UserValue[T]) -> Variable[T]: ...
@overload
def select[V](pred: Bool, on_true: V, on_false: V) -> V: ...


def select[V](pred: Bool, on_true: V, on_false: V) -> V:
    pred_ = _get(pred)

    true_vars, true_tree = pytree.flatten(cast_unchecked(on_true))
    false_vars, false_tree = pytree.flatten(cast_unchecked(on_false))

    if true_tree != false_tree:
        raise ValueError(f"incompatible types: {true_tree} and {false_tree}")

    assert len(true_vars) == len(false_vars)

    ans_vars: list[Any] = []

    for x, y in zip(true_vars, false_vars):
        x_ = _get(x)
        y_ = _get(y)
        typ = promote_types(get_type(x_), get_type(y_))
        ans_vars.append(Variable._from_val_ro(Select(typ).call(pred_, x_, y_)))

    return cast_unchecked(true_tree.unflatten(ans_vars))


################################################################################


def asm(opcode: str, outputs: Sequence[Variable[Any]], /, *args: UserValue) -> None:
    """
    this feature is in development and may produce incorret output without warning
    """
    args_ = [_get(x) for x in args]

    instr = RawInstr(
        opcode=opcode,
        in_types=TypeList(get_types(*args_)),
        out_types=TypeList(x._get_type() for x in outputs),
        continues=True,
        _reads=[EffectExternal()],
        _writes=[EffectExternal()],
        jumps=True,
    )
    out_vars = instr.emit(*args_)
    assert len(out_vars) == len(outputs)
    for out_v, uvar in zip(out_vars, outputs):
        uvar._inner.write(out_v)


def asm_fn(opcode: str, /, *args: UserValue) -> Variable[Any]:
    """
    this feature is in development and may produce incorret output without warning
    """
    out_var = Variable(AnyType)
    asm(opcode, [out_var], *args)
    return Variable(out_var)


AsmBlockLine = tuple[str, *tuple[UserValue, ...]]


def asm_block(*lines: AsmBlockLine) -> None:
    """
    this feature is in development and may produce incorret output without warning

    it is NOT allowed to jump out of the block
    """
    mvars: OrderedSet[MVar] = OrderedSet(())

    for _opcode, *args in lines:
        for v in args:
            if _get_type(v) == Label:
                raise TypeError(f"using label {v} as argument to asm_block is unsupported")
            if isinstance(v, Variable) and not v._read_only:
                mvars.add(v._inner)

    arg_vals: OrderedSet[Value] = OrderedSet(())

    def handle_arg(v: UserValue) -> int:
        if isinstance(v, Variable) and not v._read_only:
            return mvars.index(v._inner)
        return arg_vals.add(_get(v)) + len(mvars)

    linespecs: list[tuple[str, tuple[int, ...]]] = []

    for opcode, *args in lines:
        linespecs.append((opcode, tuple(handle_arg(x) for x in args)))

    in_vars: list[Var] = []
    move_instrs: list[BoundInstr[Move]] = []

    for mv in mvars:
        val = mv.read(allow_undef=True)
        (move_var,), move_instr = Move(val.type).create_bind(val)
        in_vars.append(move_var)
        move_instrs.append(move_instr)

    out_vars, inner_instr = AsmBlockInner(
        lines=linespecs,
        in_types=TypeList([mv.type for mv in mvars] + [get_type(x) for x in arg_vals]),
        out_types=TypeList(mv.type for mv in mvars),
    ).create_bind(*in_vars, *arg_vals)

    with track_caller():
        AsmBlock.from_parts(*move_instrs, inner_instr).emit()

    assert len(out_vars) == len(mvars)
    for mv, v in zip(mvars, out_vars):
        mv.write(v)
