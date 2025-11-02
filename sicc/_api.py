from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Never
from typing import Protocol
from typing import Self
from typing import Sequence
from typing import TypedDict
from typing import Unpack
from typing import overload
from typing import override

from ._core import AnyType
from ._core import AsRawCtx
from ._core import InstrBase
from ._core import InstrTypedWithArgs_api
from ._core import Label
from ._core import LabelLike
from ._core import MVar
from ._core import RawText
from ._core import TypeList
from ._core import Undef
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VirtualConst
from ._core import can_cast_implicit
from ._core import can_cast_implicit_many
from ._core import can_cast_implicit_many_or_err
from ._core import get_type
from ._core import get_types
from ._core import nan
from ._core import promote_types
from ._diagnostic import Warnings
from ._diagnostic import must_use
from ._diagnostic import register_exclusion
from ._diagnostic import suppress_warnings
from ._diagnostic import track_caller
from ._instructions import XORB
from ._instructions import XORI
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
from ._instructions import LShift
from ._instructions import Max
from ._instructions import Min
from ._instructions import Mod
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
from ._instructions import RShiftSigned
from ._instructions import RShiftUnsigned
from ._instructions import SubF
from ._instructions import SubI
from ._instructions import Transmute
from ._instructions import UnreachableChecked
from ._instructions import branch as _branch
from ._tracing import ensure_label
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import mk_mvar
from ._utils import cast_unchecked
from ._utils import isinst
from ._utils import late_fn

register_exclusion(__file__)


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

ValLabelLike = UserValue[Label] | str


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
    __xor__ = late_fn(lambda: xor)
    __rxor__ = late_fn(lambda: xor)

    __lshift__ = late_fn(lambda: lshift)
    __rshift__ = late_fn(lambda: rshift_signed)

    __mod__ = late_fn(lambda: mod)
    __rmod__ = late_fn(lambda: mod._rev_same_sig())

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
        return transmute(self, out_type)


def read_uservalue[T: VarT](v: UserValue[T]) -> Value[T]:
    if isinstance(v, VarT):
        return v
    return v._read()


def _get_label(l: ValLabelLike) -> Value[Label]:
    if isinstance(l, str):
        return ensure_label(l)
    return read_uservalue(l)


def label_ref(l: str | None = None, *, unique: bool = False) -> Label:
    """make a label to be emitted later"""
    return ensure_label(l, unique=unique)


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
    _scoped: bool


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
        scoped = kwargs.get("_scoped", False)

        self._read_only = read_only

        if mvar:
            assert mvar.type == init_or_typ
            self._inner = mvar

        elif init is not None:
            assert isinstance(init_or_typ, type)
            self._inner = mk_mvar(init_or_typ, scoped=scoped)
            self._inner.write(read_uservalue(init))

        elif isinstance(init_or_typ, type):
            self._inner = mk_mvar(init_or_typ, scoped=scoped)

        else:
            x_val = read_uservalue(init_or_typ)
            typ = get_type(x_val)
            if not read_only and typ == AnyType:
                raise TypeError("cannot infer type; specify one explicitly")
            self._inner = mk_mvar(typ, scoped=scoped)
            self._inner.write(x_val)

    def __repr__(self) -> str:
        return repr(self._inner)

    @staticmethod
    def _from_val_ro[T1: VarT](v: Value[T1], typ: type[T1] | None = None) -> VarRead[T1]:
        ans = Variable(typ or get_type(v), _read_only=True, _scoped=True)
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
        self._inner.write(read_uservalue(v))

    @staticmethod
    def undef[T1: VarT](typ: type[T1]) -> Variable[T1]:
        return Variable(undef(typ))


@dataclass(frozen=True)
class Constant[T: VarT](VarRead[T]):
    val: VirtualConst[T]

    def __repr__(self):
        return repr(self.val)

    @override
    def _get_type(self) -> type[T]:
        return self.val.get_type()

    @override
    def _read(self) -> Value[T]:
        return self.val


class _FunctionProto[T_co]:
    def _instrs_type_helper(self) -> T_co: ...


class Function[*Ts](_FunctionProto[tuple[*Ts]]):
    """
    a possibly overloaded function that maps to a instruction
    """

    _instrs: tuple[InstrBase, ...]

    def __init__(self, *overloads: *Ts) -> None:
        for x in overloads:
            assert isinstance(x, InstrBase)
        self._instrs = cast_unchecked(overloads)

    def _instrs_type_helper(self) -> tuple[*Ts]:
        return cast_unchecked(self._instrs)

    def __repr__(self) -> str:
        return "Function(" + ", ".join(type(x).__name__ for x in self._instrs) + ")"

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...
    @overload
    def __get__[V](self, obj: V, objtype: Any) -> _BoundFunction[V, tuple[*Ts]]: ...

    def __get__(self, obj: Any, objtype: Any) -> Any:
        """
        typing helper to get type checker accept Function as a method
        for methods to actually work one must use for ex late_fn
        """
        assert False

    @overload
    def __call__[*I, O](
        self: _FunctionProto[tuple[InstrTypedWithArgs_api[tuple[*I], O], *tuple[Any, ...]]],
        *args: *I,
    ) -> O: ...
    @overload
    def __call__[*I, O](
        self: _FunctionProto[tuple[Any, InstrTypedWithArgs_api[tuple[*I], O], *tuple[Any, ...]]],
        *args: *I,
    ) -> O: ...

    def __call__(self, *args: Any) -> Any:
        arg_types = tuple(_get_type(x) for x in args)

        for instr in self._instrs:
            assert isinstance(instr, InstrBase)
            if not can_cast_implicit_many(arg_types, instr.in_types):
                continue

            argvals = tuple(read_uservalue(x) for x in args)
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
            return self.call(a2, a1)  # pyright: ignore

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


def undef[T: VarT](typ: type[T] = AnyType) -> VarRead[T]:
    """return the undef constant"""
    return Constant(Undef(typ))


################################################################################

add = Function(AddI(), AddF())
sub = Function(SubI(), SubF())
mul = Function(MulI(), MulF())
div = Function(DivF())
mod = Function(Mod())

and_ = Function(AndB(), AndI())
or_ = Function(OrB(), OrI())
xor = Function(XORB(), XORI())

lshift = Function(LShift())
rshift_signed = Function(RShiftSigned())
rshift_unsigned = Function(RShiftUnsigned())

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


def unreachable_checked() -> None:
    with suppress_warnings(Warnings.Unused):
        Function(UnreachableChecked()).call()


def black_box[T: VarT](v: UserValue[T]) -> VarRead[T]:
    return Function(BlackBox(_get_type(v))).call(v)


def transmute[O: VarT](v: UserValue, out_type: type[O] = AnyType) -> VarRead[O]:
    """
    currently often generate extra move instructions; might get fixed later
    """
    if can_cast_implicit(_get_type(v), out_type):
        return Variable._from_val_ro(read_uservalue(cast_unchecked(v)), typ=out_type)
    return Function(Transmute(_get_type(v), out_type)).call(v)


################################################################################


def jump(label: ValLabelLike) -> None:
    return Jump().call(_get_label(label))


def branch(cond: Bool, on_true: ValLabelLike, on_false: ValLabelLike) -> None:
    return _branch(read_uservalue(cond), _get_label(on_true), _get_label(on_false))


def cjump(cond: Bool, on_true: ValLabelLike) -> None:
    cont = mk_internal_label("cjump_cont")
    _branch(read_uservalue(cond), _get_label(on_true), cont)
    label(cont)


################################################################################


def asm(opcode: str, outputs: Sequence[Variable[Any]], /, *args: UserValue) -> None:
    """
    this feature is in development and may produce incorret output without warning
    """
    for x in outputs:
        assert isinstance(x, Variable)
        assert not x._read_only

    args_ = [read_uservalue(x) for x in args]

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
        inputs.append(read_uservalue(v))
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


class EnumEx(Enum):
    def as_raw(self, ctx: AsRawCtx) -> RawText:
        return RawText.str(repr(self.value))
