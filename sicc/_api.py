from __future__ import annotations

import abc
import functools
import inspect
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Concatenate
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

from rich.text import Text

from . import _functions as _f
from ._core import AnyType
from ._core import Comment
from ._core import InstrBase
from ._core import InstrTypedWithArgs_api
from ._core import Label
from ._core import LabelLike
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
from ._core import nan
from ._core import promote_types
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import describe_fn
from ._diagnostic import must_use
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._instructions import AbsF
from ._instructions import AbsI
from ._instructions import AddF
from ._instructions import AddI
from ._instructions import AndB
from ._instructions import AndI
from ._instructions import AsmBlock
from ._instructions import BlackBox
from ._instructions import DivF
from ._instructions import EffectExternal
from ._instructions import Jump
from ._instructions import Max
from ._instructions import Min
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
from ._instructions import Transmute
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

    # TODO: its possible to move Function up before VarRead and
    # then get rid of all the late_fn here?

    @abc.abstractmethod
    def _read(self) -> Value[T]: ...

    @abc.abstractmethod
    def _get_type(self) -> type[T]: ...

    def __check_co(self) -> VarRead[VarT]:  # pyright: ignore[reportUnusedFunction]
        return self

    def __bool__(self) -> Never:
        err = TypeError("not possible to convert a runtime value to a compile-time bool.")
        err.add_note("use tilde operator '~' to invert a boolean")
        raise err

    # explicit @overload works with protocol requiring __add__ (so sum(), etc)
    # maybe write explicit @overload for all?
    @overload
    def __add__(self: Int, other: Int, /) -> VarRead[int]: ...
    @overload
    def __add__(self: Float, other: Float, /) -> VarRead[float]: ...
    def __add__(self: Float, other: Float, /) -> VarRead[float]:
        return add(self, other)

    __radd__ = __add__

    __sub__ = late_fn(lambda: sub)
    __rsub__ = late_fn(lambda: sub._rev_same_sig())

    __mul__ = late_fn(lambda: mul)
    __rmul__ = late_fn(lambda: mul)

    __truediv__ = late_fn(lambda: div)
    __rtruediv__ = late_fn(lambda: div._rev_same_sig())

    __invert__ = late_fn(lambda: not_)

    __and__ = late_fn(lambda: and_)
    __rand__ = late_fn(lambda: and_)
    __or__ = late_fn(lambda: or_)
    __ror__ = late_fn(lambda: or_)

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

    def __abs__(self: VarRead[float]) -> VarRead[float]:
        return abs_.call(self)

    def is_nan(self) -> VarRead[bool]:
        return equal(self, nan)

    def transmute[O: VarT](self, out_type: type[O] = AnyType) -> VarRead[O]:
        """
        currently often generate extra move instructions; might get fixed later
        """
        return Function(Transmute(self._get_type(), out_type)).call(self)


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
    _mvar: MVar | None


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
        mvar = kwargs.get("_mvar", None)

        self._read_only = read_only

        if mvar:
            assert mvar.type == init_or_typ
            self._inner = mvar
            return

        if init is not None:
            assert isinstance(init_or_typ, type)
            self._inner = mk_mvar(init_or_typ)
            self._inner.write(_get(init))
            return

        if isinstance(init_or_typ, type):
            self._inner = mk_mvar(init_or_typ)
            return

        x_val = _get(init_or_typ)
        typ = get_type(x_val)
        if not read_only and typ == AnyType:
            raise TypeError("cannot infer type; specify one explicitly")
        self._inner = mk_mvar(typ)
        self._inner.write(x_val)

    def __repr__(self) -> str:
        return repr(self._inner)

    @staticmethod
    def _from_val_ro[T1: VarT](v: Value[T1]) -> VarRead[T1]:
        ans = Variable(get_type(v), _read_only=True)
        ans._inner.write(v)
        return ans

    @override
    def _read(self) -> Value[T]:
        return self._inner.read()

    @override
    def _get_type(self) -> type[T]:
        return self._inner.type

    @property
    def value(self) -> VarRead[T]:
        return Variable(self, _read_only=True)

    @value.setter
    def value(self, v: UserValue[T]) -> None:
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

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...
    @overload
    def __get__[V](self, obj: V, objtype: Any) -> _BoundFunction[V, Ts]: ...

    def __get__(self, obj: Any, objtype: Any) -> Any:
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
    _scope: Scope | None
    _tree: pytree.PyTreeSpec | None = None
    _vars: list[MVar] | None = None

    def __init__(self, init: T | empty_t = empty):
        self._scope = _CUR_SCOPE.get()
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
            self._tree = tree
            if self._scope:
                with _CUR_SCOPE.bind(self._scope):
                    self._vars = [mk_mvar(get_type(x)) for x in vars]
            else:
                with _CUR_SCOPE.bind_clear():
                    self._vars = [mk_mvar(get_type(x)) for x in vars]
        else:
            assert self._vars is not None
            assert tree == self._tree
            assert len(vars) == len(self._vars)
        with track_caller():
            for mv, arg in zip(self._vars, vars):
                mv.write(arg)

    @property
    def value(self) -> T:
        return self.read()

    @value.setter
    def value(self, val: T) -> None:
        self.write(val)

    def ref_mut(self) -> T:
        assert self._tree is not None
        assert self._vars is not None

        vars = [Variable(x.type, _mvar=x) for x in self._vars]
        return cast_unchecked(pytree.unflatten(self._tree, vars))


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

            with add_debug_info(DebugInfo(describe=f"return val from end of {describe_fn(fn)}")):
                return_(ans)
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
        self._subr: TracedSubr | None = None

    def __repr__(self):
        return repr(self.fn)

    @overload
    def __get__(self, obj: None, objtype: type, /) -> Self: ...
    @overload
    def __get__[T, **P, R](
        self: Subr[Callable[Concatenate[T, P], R]], obj: T, objtype: Any, /
    ) -> Callable[P, R]: ...

    def __get__(self, obj: Any, objtype: Any, /):
        if obj is None:
            return self

        def inner(*args: Any, **kwargs: Any) -> Any:
            return self(
                obj, *args, **kwargs
            )  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

        return inner

    def __call__[**P, R](self: Subr[Callable[P, R]], *args: P.args, **kwargs: P.kwargs) -> R:
        bound = inspect.signature(self.fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if self._subr is None:
            self._subr = trace_to_subr(
                self.fn, **bound.arguments
            )  # pyright: ignore[reportCallIssue]
        return self._subr.call(**bound.arguments)


@dataclass
class SubrFactory:
    inline_always: bool = False

    def __call__[F](self, fn: F) -> Subr[F]:
        if self.inline_always:
            raise NotImplementedError()
        return Subr(fn)


class SubrOpts(TypedDict, total=False):
    inline_always: bool


@overload
def subr(**kwargs: Unpack[SubrOpts]) -> SubrFactory: ...
@overload
def subr[F](func: F, /, **kwargs: Unpack[SubrOpts]) -> Subr[F]: ...


def subr[F](func: F | None = None, /, **kwargs: Unpack[SubrOpts]) -> SubrFactory | Subr[F]:
    config = SubrFactory(**kwargs)
    if func is None:
        return config
    return config(func)


def inline_subr[T](fn: Callable[[], T]) -> T:
    exit = mk_internal_label("inline_subr_end")
    ret_state: State[T] = State()

    def ret_hook(val: Any) -> None:
        ret_state.write(val)
        jump(exit)

    with _RETURN_HOOK.bind(ret_hook):
        ans = fn()
        return_(ans)
        _f.unreachable_checked()

    label(exit)
    return ret_state.value


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

min_ = Function(Min(int), Min(float))
max_ = Function(Max(int), Max(float))


def clamp[T: float](
    val: UserValue[T],
    /,
    min: UserValue[T],
    max: UserValue[T],
) -> VarRead[T]:
    ans: VarRead[float] = min_(max_(val, min), max)
    return cast_unchecked(ans)


abs_ = Function(AbsI(), AbsF())

##

not_ = Function(Not())

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


##

unreachable_checked = Function(UnreachableChecked())


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


def cjump(cond: Bool, on_true: ValLabelLike) -> None:
    cont = mk_internal_label("cjump_cont")
    _f.branch(_get(cond), _get_label(on_true), cont)
    label(cont)


def if_(cond: Bool) -> AbstractContextManager[None]:
    return trace_if(_get(cond))


def while_(cond_fn: Callable[[], Bool]) -> AbstractContextManager[None]:
    def inner():
        return _get(cond_fn())

    return trace_while(inner)


def loop() -> AbstractContextManager[None]:
    return trace_while(lambda: True)


################################################################################


type _CallableOr[T] = Callable[[], T] | T


@overload
def cond[T: VarT](
    pred: Bool, on_true: _CallableOr[UserValue[T]], on_false: _CallableOr[UserValue[T]]
) -> VarRead[T]: ...
@overload
def cond[V](pred: Bool, on_true: _CallableOr[V], on_false: _CallableOr[V]) -> V: ...


def cond(pred: Bool, on_true: Any, on_false: Any) -> Any:
    pred_ = _get(pred)

    true_l = mk_internal_label("cond_true_branch")
    false_l = mk_internal_label("cond_false_branch")
    true_l2 = mk_internal_label("cond_true_branch_2")
    false_l2 = mk_internal_label("cond_false_branch_2")
    end_l = mk_internal_label("cond_end")

    _f.branch(pred_, true_l, false_l)

    def get_val(f: _CallableOr[Any]) -> Any:
        if callable(f):
            return f()
        return f

    label(true_l)
    true_out, true_tree = pytree.flatten(get_val(on_true))
    true_vals = [_get(x) for x in true_out]
    jump(true_l2)

    label(false_l)
    false_out, false_tree = pytree.flatten(get_val(on_false))
    false_vals = [_get(x) for x in false_out]
    jump(false_l2)

    if true_tree != false_tree:
        raise ValueError(f"incompatible types: {true_tree} and {false_tree}")

    assert len(true_vals) == len(false_vals)

    ans_mvars: list[MVar] = []

    for x, y in zip(true_vals, false_vals):
        typ = promote_types(get_type(x), get_type(y))
        ans_mvars.append(mk_mvar(typ))

    label(true_l2)
    for x, mv in zip(true_vals, ans_mvars):
        mv.write(x)
    jump(end_l)

    label(false_l2)
    for x, mv in zip(false_vals, ans_mvars):
        mv.write(x)
    jump(end_l)

    label(end_l)
    return cast_unchecked(true_tree.unflatten([Variable._from_val_ro(x.read()) for x in ans_mvars]))


@overload
def select[T: VarT](pred: Bool, on_true: UserValue[T], on_false: UserValue[T]) -> VarRead[T]: ...
@overload
def select[V](pred: Bool, on_true: V, on_false: V) -> V: ...


def select[V](pred: Bool, on_true: V, on_false: V) -> V:
    pred_ = _get(pred)

    true_vars, true_tree = pytree.flatten(cast_unchecked(on_true))
    false_vars, false_tree = pytree.flatten(cast_unchecked(on_false))

    if true_tree != false_tree:
        raise ValueError(f"incompatible types: {true_tree} and {false_tree}")

    assert len(true_vars) == len(false_vars)

    ans_vars: list[VarRead[Any]] = []

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
    for x in outputs:
        assert isinstance(x, Variable)
        assert not x._read_only

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


def asm_fn(opcode: str, /, *args: UserValue) -> VarRead[Any]:
    """
    this feature is in development and may produce incorret output without warning
    """
    out_var = Variable(AnyType)
    asm(opcode, [out_var], *args)
    return out_var.value


AsmBlockLine = LabelLike | tuple[str, *tuple[UserValue, ...]]


def asm_block(*lines_: AsmBlockLine) -> None:
    """
    each argument is a instruction, which is one of:
    (1) (opcode, operands, ...)
        opcode may read all operands and write to any operand Variable that is not read-only
    (2) str | Label (not implemented yet)
    """
    for l in lines_:
        if isinstance(l, LabelLike):
            raise NotImplementedError()

    lines = [x for x in lines_ if isinstance(x, tuple)]

    inputs: list[Value] = []

    def handle_arg(v: UserValue) -> int | MVar:
        if isinstance(v, Variable) and not v._read_only:
            return v._inner
        idx = len(inputs)
        inputs.append(_get(v))
        return idx

    linespecs: list[tuple[str, tuple[MVar | int, ...]]] = []

    for opcode, *args in lines:
        linespecs.append((opcode, tuple(handle_arg(x) for x in args)))

    with track_caller():
        AsmBlock(
            lines=linespecs,
            in_types=TypeList([get_type(x) for x in inputs]),
            out_types=(),
        ).call(*inputs)
