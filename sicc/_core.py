from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Final
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Protocol
from typing import Self
from typing import TypeGuard
from typing import TypeVar
from typing import cast
from typing import overload
from typing import override
from typing import runtime_checkable

from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.console import Group
from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import catch_ex_and_exit
from ._diagnostic import check_must_use
from ._diagnostic import clear_debug_info
from ._diagnostic import debug_info
from ._diagnostic import register_exclusion
from ._diagnostic import show_pending_diagnostics
from ._utils import ByIdMixin
from ._utils import Cell
from ._utils import ReprAs
from ._utils import cast_unchecked
from ._utils import crc32
from ._utils import disjoint_union
from ._utils import get_id
from ._utils import narrow_unchecked
from ._utils import safe_cast
from .config import verbose

if TYPE_CHECKING:
    from ._api import UserValue
    from ._api import VarRead
    from ._instructions import Bundle
    from ._instructions import EmitLabel
    from ._instructions import RawInstr


register_exclusion(__file__)


class Register(Enum):
    R0 = "r0"
    R1 = "r1"
    R2 = "r2"
    R3 = "r3"
    R4 = "r4"
    R5 = "r5"
    R6 = "r6"
    R7 = "r7"
    R8 = "r8"
    R9 = "r9"
    R10 = "r10"
    R11 = "r11"
    R12 = "r12"
    R13 = "r13"
    R14 = "r14"
    R15 = "r15"
    R16 = "r16"

    RA = "ra"
    SP = "sp"


@dataclass(frozen=True)
class RegInfo:
    allocated_reg: Register | None = None
    force_reg: Register | None = None
    preferred_reg: Register | None = None
    preferred_weight: int = 0

    @property
    def allocated(self) -> Register:
        assert self.allocated_reg is not None
        return self.allocated_reg


@dataclass(frozen=True)
class RegallocSkip:
    """allow a output var to be skipped if it is not used anywhere"""

    v: Var


@dataclass(frozen=True)
class RegallocTie:
    """v1 and v2 should try to be allocated the same reg"""

    v1: Var | MVar
    v2: Var | MVar
    force: bool

    def normalize(self) -> RegallocTie:
        v1, v2 = sorted([self.v1, self.v2])
        return RegallocTie(v1, v2, self.force)


@dataclass(frozen=True)
class RegallocExtend:
    """
    by default, a input reg can be reused as a output.
    this marks a input reg as not the case
    """

    v: Var


RegallocPref = RegallocSkip | RegallocTie | RegallocExtend


@dataclass
class RawText:
    """part of final assembly"""

    text: Text

    @staticmethod
    def str(s: str, style: str = "ic10.other") -> RawText:
        return RawText(Text(s, style))


@dataclass
class AsRawCtx:
    instr: BoundInstr[RawInstr] | None
    linenums: LineNums


@runtime_checkable
class AsRaw(Protocol):
    def as_raw(self, ctx: AsRawCtx) -> RawText: ...


class AnyType_(AsRaw):
    def as_raw(self, ctx: AsRawCtx):
        assert False, "unreachable"


AnyType: type[Any] = AnyType_


@dataclass(frozen=True)
class Undef(AsRaw):
    @staticmethod
    def undef() -> Any:
        return Undef()

    def as_raw(self, ctx: AsRawCtx) -> RawText:
        return RawText.str("0", "ic10.undef")

    def __repr__(self) -> str:
        return "undef"


@dataclass(frozen=True)
class PinType(AsRaw):
    name: str

    def as_raw(self, ctx: AsRawCtx) -> RawText:
        raise NotImplementedError()


nan: float = math.nan

type VarTS = tuple[type[VarT], ...]


T_co = TypeVar("T_co", covariant=True, bound="VarT", default="VarT")


@dataclass(frozen=True, eq=False)
class Var(Generic[T_co], ByIdMixin):
    """
    internal varaible used by the compiler.
    unlike Variable, equality and comparison is by variable id.

    A Var must always be assigned in one place. A MVar may be assigned any number of times.

    currently only created with the mk_var function in _tracing.py (TODO: is this true?)
    """

    type: type[T_co]
    id: int
    #: internal check. user invalid uses should be caught when the Var is found not in _CUR_SCOPE
    #: TODO: not used rn
    live: Cell[bool]
    reg: RegInfo

    debug: DebugInfo

    def check_type[T: VarT](self, typ: type[T]) -> Var[T]:
        can_cast_implicit_or_err(self.type, typ)
        return cast_unchecked(self)

    def __repr__(self) -> str:
        if self.reg.allocated_reg:
            return f"%{self.reg.allocated_reg.value}"
        return f"%v{self.id}"


@dataclass(frozen=True, eq=False)
class Label(ByIdMixin):
    id: str
    debug: DebugInfo

    #: Non-implicit labels are generally user created
    #: they are always kept
    implicit: bool

    def __repr__(self) -> str:
        return self.id


@runtime_checkable
class EffectBase(Protocol):
    """
    represents a location that side-effectful statements changes.

    subclass must have hash and equality
    """

    # return True if both read and write opts on self and other can be reordered
    def known_distinct(self, other: Self) -> bool:
        return False


type EffectRes = EffectBase | Iterable[EffectBase] | None


def known_distinct(x: EffectBase, y: EffectBase) -> bool:
    """
    whether self and other are completely unrelated.

    if returns true, both read and write operation on self may be reordered with operation on other
    """
    tx = type(x)
    ty = type(y)

    common_base = (set(tx.__mro__) & set(ty.__mro__)) - set(safe_cast(type, EffectBase).__mro__)

    if len(common_base) == 0:
        return True

    # call the method on the more specific class
    if issubclass(tx, ty):  # tx is child
        return x.known_distinct(y)
    if issubclass(ty, tx):
        return y.known_distinct(x)

    return False


VarT = bool | int | float | str | Label | AsRaw

LabelLike = Label | str

type Value[T: VarT = VarT] = Var[T] | T

InteralBool = Value[bool]
InternalInt = Value[int]
InternalFloat = Value[float]
InternalValLabel = Value[Label]


def get_type[T: VarT](x: Value[T]) -> type[T]:
    if isinstance(x, Var):
        return x.type
    else:
        return type(x)


def get_types(*xs: Value) -> VarTS:
    return tuple(get_type(x) for x in xs)


def can_cast_implicit(t1: type[VarT], t2: type[VarT]) -> bool:
    """
    this must be kept transitive; caller may depend on this
    """
    if t1 == PinType:
        t1 = int
    if t2 == PinType:
        t2 = int

    if t1 == t2:
        return True

    if t1 == AnyType:
        return True
    if t1 == Undef:
        return True

    if (t1, t2) == (int, float):
        return True
    if (t1, t2) == (bool, int):
        return True
    if (t1, t2) == (bool, float):
        return True

    return False


def can_cast_implicit_or_err(t1: type[VarT], t2: type[VarT]) -> None:
    if not can_cast_implicit(t1, t2):
        raise TypeError(f"not possible to use {t1} as {t2}")


def promote_types(t1: type[VarT], t2: type[VarT]) -> type[VarT]:
    if t1 == t2:
        return t1

    for target in [bool, int, float]:
        if can_cast_implicit(t1, target) and can_cast_implicit(t2, target):
            return target

    raise TypeError(f"incompatible types: {t1} and {t2}")


def can_cast_implicit_many(t1: VarTS, t2: VarTS) -> bool:
    if len(t1) != len(t2):
        return False
    for x, y in zip(t1, t2):
        if not can_cast_implicit(x, y):
            return False
    return True


def can_cast_implicit_many_or_err(t1: VarTS, t2: VarTS) -> None:
    if not can_cast_implicit_many(t1, t2):
        t1_ = tuple(ReprAs(x.__name__) for x in t1)
        t2_ = tuple(ReprAs(x.__name__) for x in t2)
        raise TypeError(f"not possible to use {t1_} as {t2_}")


def can_cast_val[T: VarT](v: Value, typ: type[T]) -> TypeGuard[Value[T]]:
    return can_cast_implicit(get_type(v), typ)


class TypeList[T](tuple[type[VarT], ...]):
    def _inv_marker(self, x: T) -> T: ...


class _InstrTypedIn[I](Protocol):
    in_types: Final[I]


class _InstrTypedOut[O](Protocol):
    out_types: Final[O]


class InstrTypedWithArgsIn[I](Protocol):
    def _static_in_typing_helper(self, x: I, /) -> None: ...


class InstrTypedWithArgsOut[O](Protocol):
    def _static_out_typing_helper(self) -> O: ...


class InstrTypedWithArgs[I, O, S = InstrBase](Protocol):
    def _self(self) -> S: ...
    def _static_in_typing_helper(self, x: I, /) -> None: ...
    def _static_out_typing_helper(self) -> O: ...


class InstrTypedWithArgs_api[I, O](Protocol):
    def _static_in_typing_helper_api(self, x: I, /) -> None: ...
    def _static_out_typing_helper_api(self) -> O: ...


def _default_annotate(x: BoundInstr) -> str:
    return ""


FORMAT_SCOPE_CONTEXT: Cell[Scope] = Cell()
FORMAT_ANNOTATE: Cell[Callable[[BoundInstr], str | Text]] = Cell(_default_annotate)
FORMAT_VAL_FN: Cell[Callable[[Value | MVar], str]] = Cell(repr)
FORMAT_LINENUMS: Cell[LineNums] = Cell()


def get_style(typ: type[VarT]) -> str:
    if issubclass(typ, (bool, int, float, str)):
        return "ic10." + typ.__name__
    if issubclass(typ, Label):
        return "ic10.label"
    if issubclass(typ, Undef):
        return "ic10.undef"
    return "ic10.other"


def format_val(v: Value) -> Text:
    typ = get_type(v)
    if issubclass(typ, Label):
        priv = (f := FORMAT_SCOPE_CONTEXT.get()) and (v in f.private_labels)
        return Text(FORMAT_VAL_FN(v), "ic10.label_private" if priv else "ic10.label")

    return Text(FORMAT_VAL_FN(v), get_style(typ))


LowerRes = tuple[Var, ...] | Var | None


class UnpackPolicy(Protocol):
    def should_unpack(self, instr: BoundInstr, /) -> bool: ...


@dataclass(frozen=True)
class AlwaysUnpack(UnpackPolicy):
    def should_unpack(self, instr: BoundInstr) -> bool:
        return True


@dataclass(frozen=True)
class NeverUnpack(UnpackPolicy):
    def should_unpack(self, instr: BoundInstr) -> bool:
        return False


class InstrBase(abc.ABC):
    # required overrides
    in_types: VarTS
    out_types: VarTS

    def lower(self, instr: BoundInstr[Any], /) -> Iterable[BoundInstr]:
        """
        The final step before outputing asm: Convert instr to RawInstr

        If returning other instrs, the lower method on those will be called until
        everthing is a RawInstr or EmitLabel

        this is the only place where the resulting fragment would be invalid.
        we still have this return boundinstr to make use of the existing formatting stack.

        one should not pass the resulting fragment to any analysis passes
        """
        raise NotImplementedError(f"not implemented, or {type(self)} is not supposed to be lowered")

    # optional overrides

    #: continues to the next instruction
    continues: bool = True

    def get_continues(self, instr: BoundInstr[Any], /) -> bool:
        """if overriding this, "continues" variable have no effect"""
        return self.continues

    #: if True, may jump to any label that is a input
    #: if False, never jumps
    jumps: bool = True

    def jumps_to(self, instr: BoundInstr[Any], /) -> Iterable[InternalValLabel]:
        """if overriding this, "jumps" variable have no effect"""
        if self.jumps:
            for xt, x in zip(instr.check_type().instr.in_types, instr.inputs):
                if xt == Label:
                    assert can_cast_val(x, Label)
                    yield x

    # impure operations MUST override reads and/or writes
    # writes does NOT imply reads
    def reads(self, instr: BoundInstr[Any], /) -> EffectRes:
        return None

    def writes(self, instr: BoundInstr[Any], /) -> EffectRes:
        if len(self.out_types) == 0 and len(instr.jumps_to()) == 0 and instr.continues:
            raise NotImplementedError(f"{type(self)} returns nothing, so it must override 'writes'")

    def defines_labels(self, instr: BoundInstr[Any], /) -> Iterable[Label]:
        """does it contain EmitLabel or equirvalent?"""
        return []

    def regalloc_prefs(self, instr: BoundInstr[Any], /) -> Iterable[RegallocPref]:
        """
        FIXME:
        currently RegallocSkip are accessed not-unpacked while RegallocTie are accessed as always-unpacked
        """
        return []

    def format_expr_part(self, instr: BoundInstr[Any], /) -> Text:
        ans = Text()
        mark = self.jumps and any(get_type(x) == Label for x in instr.inputs)
        mark |= not instr.continues
        ans.append(type(self).__name__, "ic10.jump" if mark else "ic10.opcode")
        for x in instr.inputs:
            ans += " "
            ans += format_val(x)
        return ans

    def format(self, instr: BoundInstr[Any], /) -> Text:
        expr_part = self.format_expr_part(instr)

        ans = Text()
        if len(instr.outputs) > 0:
            ans += format_val(instr.outputs[0])
            for x in instr.outputs[1:]:
                ans += ", "
                ans += format_val(x)
            ans += " = "
        ans += expr_part
        return ans

    def format_comment(self, instr: BoundInstr[Any], /) -> Text:
        comment = Text()
        if loc_info := instr.debug.location_info_brief():
            comment.append("  # " + loc_info, "ic10.comment")
        if annotation := FORMAT_ANNOTATE.value(instr):
            comment.append("  # ", "ic10.comment")
            comment.append(annotation)
        return comment

    def format_with_anno(self, instr: BoundInstr[Any], /) -> RenderableType:
        from ._instructions import EmitLabel

        ans = Text()

        if linenums := FORMAT_LINENUMS.get():
            if (line := linenums.instr_lines.get(instr)) is not None and not instr.isinst(
                EmitLabel
            ):
                prefix = repr(line).rjust(3) + ": "
            else:
                prefix = " " * 5

            ans += Text(prefix, "ic10.linenum")

        ans += self.format(instr)
        ans += self.format_comment(instr)

        return ans

    ################################################################################
    # stub methods for static typing
    ################################################################################

    def _self(self) -> Self:
        return self

    @overload
    def _static_out_typing_helper(self: _InstrTypedOut[tuple[()]]) -> tuple[()]: ...
    @overload
    def _static_out_typing_helper[A: VarT](
        self: _InstrTypedOut[tuple[type[A]]],
    ) -> tuple[Var[A]]: ...
    @overload
    def _static_out_typing_helper[T](self: _InstrTypedOut[TypeList[T]]) -> T: ...
    def _static_out_typing_helper(self: Any) -> Any: ...

    @overload
    def _static_in_typing_helper(self: _InstrTypedIn[tuple[()]], x: tuple[()], /) -> None: ...
    @overload
    def _static_in_typing_helper[A: VarT](
        self: _InstrTypedIn[tuple[type[A]]], x: tuple[Value[A]], /
    ) -> None: ...
    @overload
    def _static_in_typing_helper[A: VarT, B: VarT](
        self: _InstrTypedIn[tuple[type[A], type[B]]], x: tuple[Value[A], Value[B]], /
    ) -> None: ...
    @overload
    def _static_in_typing_helper[A: VarT, B: VarT, C: VarT](
        self: _InstrTypedIn[tuple[type[A], type[B], type[C]]],
        x: tuple[Value[A], Value[B], Value[C]],
        /,
    ) -> None: ...
    @overload
    def _static_in_typing_helper[T](self: _InstrTypedIn[TypeList[T]], x: T, /) -> None: ...
    def _static_in_typing_helper(self: Any, x: Any, /) -> None: ...

    ################################################################################
    # equivalents, for use with the Variable class in _api.py
    ################################################################################

    @overload
    def _static_out_typing_helper_api(self: _InstrTypedOut[tuple[()]]) -> None: ...
    @overload
    def _static_out_typing_helper_api[A: VarT](
        self: _InstrTypedOut[tuple[type[A]]],
    ) -> VarRead[A]: ...
    def _static_out_typing_helper_api(self: Any) -> Any: ...

    @overload
    def _static_in_typing_helper_api(self: _InstrTypedIn[tuple[()]], x: tuple[()], /) -> None: ...
    @overload
    def _static_in_typing_helper_api[A: VarT](
        self: _InstrTypedIn[tuple[type[A]]], x: tuple[UserValue[A]], /
    ) -> None: ...
    @overload
    def _static_in_typing_helper_api[A: VarT, B: VarT](
        self: _InstrTypedIn[tuple[type[A], type[B]]], x: tuple[UserValue[A], UserValue[B]], /
    ) -> None: ...
    @overload
    def _static_in_typing_helper_api[A: VarT, B: VarT, C: VarT](
        self: _InstrTypedIn[tuple[type[A], type[B], type[C]]],
        x: tuple[UserValue[A], UserValue[B], UserValue[C]],
        /,
    ) -> None: ...
    @overload
    def _static_in_typing_helper_api[A: VarT, B: VarT, C: VarT, D: VarT](
        self: _InstrTypedIn[tuple[type[A], type[B], type[C], type[D]]],
        x: tuple[UserValue[A], UserValue[B], UserValue[C], UserValue[D]],
        /,
    ) -> None: ...
    def _static_in_typing_helper_api(self: Any, x: Any, /) -> None: ...

    ################################################################################

    def check_inputs(self, *args: Value) -> None:
        for x in args:
            _ck_val(x)
        in_types = self.in_types
        arg_types = get_types(*args)
        can_cast_implicit_many_or_err(arg_types, in_types)

    def check_outputs(self, *args: Var) -> None:
        for x in args:
            _ck_val(x)
        out_types = self.out_types
        arg_types = get_types(*args)
        can_cast_implicit_many_or_err(out_types, arg_types)

    def bind[*I, O, S](
        self: InstrTypedWithArgs[tuple[*I], O, S], out_vars: O, /, *args: *I
    ) -> BoundInstr[S]:
        return self.bind_untyped(out_vars, *args)  # pyright: ignore

    def bind_untyped(self, out_vars: tuple[Var, ...], /, *args: Value) -> BoundInstr[Self]:
        self.check_inputs(*args)
        self.check_outputs(*out_vars)
        return BoundInstr((get_id(),), self, args, out_vars, debug_info())

    def create_bind[*I, O, S](
        self: InstrTypedWithArgs[tuple[*I], O, S], *args: *I
    ) -> tuple[O, BoundInstr[S]]:
        from ._tracing import mk_var

        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)
            assert narrow_unchecked(args, tuple[Value, ...])

        self.check_inputs(*args)
        out_vars = tuple(mk_var(x) for x in self.out_types)

        ans = BoundInstr((get_id(),), self._self(), args, out_vars, debug_info())
        return cast_unchecked(out_vars), ans

    def emit[*I, O](self: InstrTypedWithArgs[tuple[*I], O], *args: *I) -> O:
        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)

        out_vars, bound = self.create_bind(*args)
        bound.emit()
        return out_vars

    @overload
    def call[*I, A](self: InstrTypedWithArgs[tuple[*I], tuple[A]], *args: *I) -> A: ...
    @overload
    def call[*I](self: InstrTypedWithArgs[tuple[*I], tuple[()]], *args: *I) -> None: ...

    def call(self: InstrTypedWithArgs[Any, Any], *args: Any) -> Any:
        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)
        ans = self.emit(*args)
        if len(ans) == 0:
            return None
        if len(ans) == 1:
            return ans[0]
        assert False


def _ck_val(v: Value) -> None:
    from ._tracing import ck_val

    return ck_val(v)


B_co = TypeVar("B_co", covariant=True, default=InstrBase)


@dataclass(frozen=True, eq=False)
class BoundInstr(Generic[B_co], ByIdMixin):
    id: tuple[int, ...]
    instr: B_co
    inputs: tuple[Value, ...]
    outputs: tuple[Var, ...]
    debug: DebugInfo

    # TODO: organize these methods

    @property
    def inputs_[I1](self: BoundInstr[InstrTypedWithArgsIn[I1]]) -> I1:
        return cast_unchecked(self.inputs)

    @property
    def outputs_[O1](self: BoundInstr[InstrTypedWithArgsOut[O1]]) -> O1:
        return cast_unchecked(self.outputs)

    def check_scope(self):
        for x in self.inputs:
            _ck_val(x)
        for x in self.outputs:
            _ck_val(x)

    def unpack_untyped(self: BoundInstr[Bundle[*tuple[Any, ...]]]) -> tuple[BoundInstr, ...]:
        return self.instr.parts(self)

    @overload
    def unpack[*Ts](self: BoundInstr[Bundle[*Ts]]) -> tuple[*Ts]: ...
    @overload
    def unpack(self) -> tuple[BoundInstr, ...] | None: ...

    def unpack(self) -> Any:
        from ._instructions import Bundle

        if i := self.isinst(Bundle):
            return i.instr.parts(i)
        return None

    def unpack_rec(
        self: BoundInstr, policy: UnpackPolicy = AlwaysUnpack()
    ) -> list[BoundInstr] | None:
        if policy.should_unpack(self) and (parts := self.unpack()) is not None:
            return [p for x in parts for p in x.unpack_rec_or_self(policy)]
        return None

    def unpack_rec_or_self(
        self: BoundInstr, policy: UnpackPolicy = AlwaysUnpack()
    ) -> list[BoundInstr]:
        if (unpacked := self.unpack_rec(policy)) is not None:
            return unpacked
        return [self]

    @property
    def continues(self: BoundInstr) -> bool:
        return self.instr.get_continues(self)

    def jumps_to(self: BoundInstr) -> list[InternalValLabel]:
        return list(self.instr.jumps_to(self))

    def __rich__(self) -> RenderableType:
        if TYPE_CHECKING:
            assert narrow_unchecked(self, BoundInstr)
        return self.instr.format_with_anno(self)

    def __repr__(self):
        if TYPE_CHECKING:
            assert narrow_unchecked(self, BoundInstr)
        return repr(self.instr.format(self).plain)

    def isinst[T](self, instr_type: type[T] | tuple[type[T], ...]) -> BoundInstr[T] | None:
        if isinstance(self.instr, instr_type):
            return cast_unchecked(self)
        return None

    def check_type[T](self, instr_type: type[T] = InstrBase) -> BoundInstr[T]:
        assert isinstance(self.instr, instr_type)
        self.check_scope()
        return cast_unchecked(self)

    def emit(self) -> None:
        from ._tracing import emit_bound

        emit_bound(self.check_type())

    def sub_val(
        self, v: Value, rep: Value, inputs: bool = False, outputs: bool = False, strict: bool = True
    ) -> BoundInstr[B_co]:
        assert inputs or outputs
        if inputs:
            if strict:
                assert v in self.inputs
            new_inputs = tuple(rep if v == x else x for x in self.inputs)
        else:
            new_inputs = self.inputs
        if outputs:
            assert isinstance(v, Var)
            assert isinstance(rep, Var)
            if strict:
                assert v in self.outputs
            new_outputs = tuple(rep if v == x else x for x in self.outputs)
        else:
            new_outputs = self.outputs

        return BoundInstr((get_id(),), self.instr, new_inputs, new_outputs, self.debug)

    def reads(self) -> list[EffectBase]:
        assert isinstance(self.instr, InstrBase)
        ans = self.instr.reads(self)
        if ans is None:
            return []
        if isinstance(ans, EffectBase):
            return [ans]
        return list(ans)

    def writes(self) -> list[EffectBase]:
        assert isinstance(self.instr, InstrBase)
        ans = self.instr.writes(self)
        if ans is None:
            return []
        if isinstance(ans, EffectBase):
            return [ans]
        return list(ans)

    def defines_labels(self: BoundInstr) -> list[Label]:
        return list(self.instr.defines_labels(self))

    def is_pure(self) -> bool:
        return (not self.reads()) and (not self.writes())

    def is_side_effect_free(self) -> bool:
        return not self.writes()

    def regalloc_prefs(self) -> list[RegallocPref]:
        assert isinstance(self.instr, InstrBase)
        return list(self.instr.regalloc_prefs(self))


################################################################################


@dataclass(frozen=True)
class EffectComment(EffectBase):
    pass


class Comment(InstrBase):
    def __init__(self, text: Text, *arg_types: type[VarT]) -> None:
        self.text = text
        self.in_types = cast(TypeList[tuple[Value, ...]], TypeList(arg_types))
        self.out_types = ()

    @override
    def format(self, instr: BoundInstr[Self]) -> Text:
        ans = Text()
        ans += Text("*", "ic10.jump")
        ans += " "
        ans += self.text
        for x in instr.inputs:
            ans += " "
            ans += format_val(x)
        return ans

    # dont reorder comments
    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectComment()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectComment()


################################################################################


@dataclass(frozen=True, eq=False)
class MVar[T: VarT = Any](ByIdMixin):
    """
    currently only created with the mk_mvar function in _tracing.py (TODO: is this true?)
    """

    type: type[T]
    id: int
    reg: RegInfo
    debug: DebugInfo

    def read(self, allow_undef: bool = False) -> Var[T]:
        return ReadMVar(self, allow_undef=allow_undef).call()

    def write(self, v: Value[T]) -> None:
        () = WriteMVar(self).emit(v)

    def __repr__(self) -> str:
        if self.reg.allocated_reg:
            return f"%{self.reg.allocated_reg.value}"
        return f"%s{self.id}"

    def _format(self):
        priv = (f := FORMAT_SCOPE_CONTEXT.get()) and (self in f.private_mvars)
        ans = Text("", "underline" if priv else "underline reverse")
        ans.append(FORMAT_VAL_FN(self), get_style(self.type))
        return Text() + ans

    def _as_var_with_reg(self) -> Var[T]:
        from ._tracing import mk_var

        assert self.reg.allocated_reg is not None
        return mk_var(self.type, reg=self.reg, debug=self.debug)


@dataclass(frozen=True)
class EffectMvar(EffectBase):
    s: MVar

    def known_distinct(self, other: Self) -> bool:
        return self.s != other.s


class ReadMVar[T: VarT = Any](InstrBase):
    s: MVar[T]

    jumps = False

    def __init__(self, s: MVar[T], allow_undef: bool) -> None:
        self.s = s
        self.in_types = ()
        self.out_types = (s.type,)
        self.allow_undef = allow_undef

    @override
    def format_expr_part(self, instr: BoundInstr[Self], /) -> Text:
        return self.s._format() + (" (allow_undef)" if self.allow_undef else "")

    @override
    def reads(self, instr: BoundInstr[Self], /) -> EffectBase:
        return EffectMvar(self.s)

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        from ._instructions import Move

        (opt,) = instr.outputs_

        if opt.reg.allocated == self.s.reg.allocated:
            return
        else:
            yield Move(self.s.type).bind((opt,), self.s._as_var_with_reg())

    @override
    def regalloc_prefs(self, instr: BoundInstr[Self]) -> Iterable[RegallocPref]:
        (out_var,) = instr.outputs_
        yield RegallocTie(instr.instr.s, out_var, force=False)


class WriteMVar[T: VarT = Any](InstrBase):
    s: MVar[T]

    jumps = False

    def __init__(self, s: MVar[T]) -> None:
        self.s = s
        self.in_types = (s.type,)
        self.out_types = ()

    @override
    def format(self, instr: BoundInstr[Self], /) -> Text:
        return self.s._format() + " = " + format_val(instr.inputs_[0])

    @override
    def writes(self, instr: BoundInstr[Any], /) -> EffectBase:
        return EffectMvar(self.s)

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        from ._instructions import Move

        (ipt,) = instr.inputs_

        if isinstance(ipt, Var) and ipt.reg.allocated == self.s.reg.allocated:
            return
        else:
            yield Move(self.s.type).bind((self.s._as_var_with_reg(),), ipt)

    @override
    def regalloc_prefs(self, instr: BoundInstr[Self]) -> Iterable[RegallocPref]:
        (arg,) = instr.inputs_
        if isinstance(arg, Var):
            yield RegallocTie(arg, instr.instr.s, force=False)


################################################################################

MapInstrsRes = BoundInstr | Iterable[BoundInstr] | None
MapInstrsFn = Callable[[BoundInstr], MapInstrsRes]


def format_instr_list(instrs: list[BoundInstr]) -> Iterator[RenderableType]:
    from ._instructions import EmitLabel

    prev_is_label = True
    prev_cont = True
    for x in instrs:
        sep = not prev_cont
        if x.isinst(EmitLabel):
            if not prev_is_label:
                sep = True
            prev_is_label = True
        else:
            prev_is_label = False
        prev_cont = x.continues

        if sep:
            yield ""
        # eagerly, so that FORMAT_CTX takes effect
        yield x.__rich__()


@dataclass
class Block:
    """
    Must start with a EmitLabel and end with something with continues=False
    """

    # maybe replace with OrderedSet?
    contents: list[BoundInstr]
    debug: DebugInfo = field(compare=False)

    @property
    def label_instr(self) -> BoundInstr[EmitLabel]:
        from ._instructions import EmitLabel

        return self.contents[0].check_type(EmitLabel)

    @property
    def label(self) -> Label:
        (ans,) = self.label_instr.inputs_
        # EmitLabel must be a constant Label
        assert isinstance(ans, Label)
        return ans

    @property
    def end(self) -> BoundInstr:
        ans = self.contents[-1]
        assert not ans.continues
        return ans

    @property
    def body(self) -> list[BoundInstr]:
        assert len(self.contents) >= 2
        return self.contents[1:-1]

    def basic_check(self):
        _ = self.label
        _ = self.end
        _ = self.body

    def __rich__(self) -> RenderableType:
        return Panel(
            Group(*format_instr_list(self.contents)),
            # title=format_val(self.label),
            # title = "Block",
            # title_align="left",
        )

    def map_instrs(self, fn: MapInstrsFn) -> None:
        def get(x: BoundInstr) -> list[BoundInstr]:
            with clear_debug_info(), add_debug_info(x.debug):
                ans = fn(x)
                if ans is None:
                    return [x]
                if isinstance(ans, BoundInstr):
                    return [ans]
                ans = list(ans)
                for y in ans:
                    assert isinstance(y, BoundInstr)

                return ans

        def gen() -> Iterator[BoundInstr]:
            for x in self.contents:
                yield from get(x)

        self.contents = list(gen())


@dataclass(eq=False)
class Scope:
    """
    items here may be dead; we dont require compiler to explicitly remove vars
    """

    #: all vars are private
    vars: OrderedSet[Var]

    #: private to this fragment
    private_mvars: OrderedSet[MVar]

    #: Labels that must not be explicity referenced outside this Fragment
    #:
    #: they may still be assigned to variables and used outside;
    #: whether that happens is inferred in compute_control_flow
    #:
    private_labels: OrderedSet[Label]

    def merge_child(self, child: Scope) -> None:
        self.vars = disjoint_union(self.vars, child.vars)
        self.private_mvars = disjoint_union(self.private_mvars, child.private_mvars)
        self.private_labels = disjoint_union(self.private_labels, child.private_labels)


@dataclass(eq=False)
class Fragment:
    """
    a set of blocks.
    Analysis and optimization generally runs on a Fragment

    Fragment is generally used mutably, unless specified otherwise

    there is no default entry or exit;
    the fragment is enterred if jumped to a tag in it, and exitted if jumpped out.

    all Var are private:
    assigned exactly once in the fragement, and must not be used/read outside.

    MVar that is not in private_mvars may be used outside.
    """

    #: declares the fragment to have completed tracing
    finished_init: bool

    #: body of the fragment
    blocks: dict[Label, Block]

    #: Var in this fragment
    scope: Scope

    def __rich__(self, title: str = "Fragment") -> RenderableType:
        def content() -> Iterator[RenderableType]:
            if not self.finished_init:
                yield "(unfinished)"
            for b in self.blocks.values():
                # eagerly, so that FORMAT_CTX takes effect
                yield b.__rich__()

        with FORMAT_SCOPE_CONTEXT.bind(self.scope):
            return Panel(
                Group(*content()),
                title=Text(title, "ic10.title"),
                title_align="left",
            )

    def basic_check(self):
        assert self.finish

        for l, b in self.blocks.items():
            assert b.label == l
            b.basic_check()

    def finish(self):
        assert not self.finished_init
        self.finished_init = True

    def merge_child(self, child: Fragment) -> None:
        assert not self.finished_init
        child.basic_check()

        blocks = self.blocks | child.blocks
        assert len(blocks) == len(self.blocks) + len(child.blocks)
        self.blocks = blocks

        self.scope.merge_child(child.scope)

    ################################################################################

    def map_instrs[F: MapInstrsFn](self, fn: F) -> F:
        ans: dict[Label, Block] = {}
        for x in self.blocks.values():
            x.map_instrs(fn)
            assert x.label not in ans
            ans[x.label] = x
        self.blocks = ans
        self.basic_check()
        return fn

    def replace_instr(self, instr: BoundInstr) -> Callable[[Callable[[], MapInstrsRes]], None]:

        def inner(fn: Callable[[], MapInstrsRes]) -> None:
            found = Cell(False)

            def inner2(x: BoundInstr) -> MapInstrsRes:
                if x == instr:
                    assert not found.value
                    found.value = True
                    return fn()

            self.map_instrs(inner2)
            assert found.value

        return inner


################################################################################


@dataclass
class LineNums:
    instr_lines: dict[BoundInstr, int]
    label_lines: dict[Label, int]


def format_raw_val(x: Value, ctx: AsRawCtx, typ: type[VarT], debug: DebugInfo) -> RawText:
    if typ == PinType:
        if isinstance(x, int):
            return RawText(Text(f"d{x}", "ic10.pin"))
        elif isinstance(x, Var):
            return RawText(Text(f"d{x.reg.allocated.value}", "ic10.pin"))
        elif isinstance(x, PinType):
            return RawText(Text(x.name, "ic10.pin"))
        else:
            raise TypeError(x)

    style = get_style(get_type(x))

    if isinstance(x, bool):
        x = 1 if x else 0
    if isinstance(x, int | float):
        if math.isnan(x):
            debug.error("unsupported nan literal").throw()
        return RawText(Text(repr(x), style))

    if isinstance(x, str):
        hash_text = Text.assemble("HASH(", ('"' + x + '"', style), ")")
        hash_num = Text(repr(crc32(x)), style)
        if len(hash_text) <= len(hash_num):
            return RawText(hash_text)
        return RawText(hash_num)

    if isinstance(x, Label):
        return RawText(Text(repr(ctx.linenums.label_lines[x]), style))

    if isinstance(x, Var):
        return RawText(Text(x.reg.allocated.value, style))

    return x.as_raw(ctx)


@dataclass
class TracedProgram:
    start: Label
    frag: Fragment

    did_optimize: bool = False
    did_regalloc: bool = False

    def check(self) -> None:
        from ._transforms import global_checks

        global_checks(self.frag)
        check_must_use()
        show_pending_diagnostics()

    def optimize(self) -> None:
        from ._transforms import global_opts

        global_opts(self.frag)

        if verbose.value >= 1:
            print("after global optimize:")
            print(self.frag)

        self.did_optimize = True

    def regalloc(self) -> None:
        """required to first run optimize"""
        from ._transforms import regalloc_and_lower
        from ._transforms.fuse_blocks import force_fuse_into_one

        assert self.did_optimize
        if self.did_regalloc:
            return

        regalloc_and_lower(self.frag)
        force_fuse_into_one(self.frag, self.start)

        self.did_regalloc = True

    def _as_instrs(self) -> list[BoundInstr]:
        assert self.did_regalloc
        assert len(self.frag.blocks) == 1
        (block,) = self.frag.blocks.values()
        return block.contents

    def _get_linenums(self) -> LineNums:
        from ._instructions import EmitLabel
        from ._instructions import EndPlaceholder
        from ._instructions import RawInstr

        instrs = self._as_instrs()

        c = 0
        instr_lines: dict[BoundInstr, int] = {}
        label_lines: dict[Label, int] = {}

        for instr in instrs:
            if (i := instr.isinst(RawInstr)) or (i := instr.isinst(Comment)):
                instr_lines[i] = c
                c += 1
            elif i := instr.isinst(EmitLabel):
                v = i.inputs_[0]
                assert not isinstance(v, Var)
                label_lines[v] = c
            elif i := instr.isinst(EndPlaceholder):
                instr_lines[i] = c
            else:
                raise TypeError(instr)

        return LineNums(instr_lines, label_lines)

    def gen_asm(self) -> RawText:
        with catch_ex_and_exit(self.frag):
            return self._gen_asm()

    def _gen_asm(self) -> RawText:
        from ._instructions import EmitLabel
        from ._instructions import EndPlaceholder
        from ._instructions import RawInstr

        linenums = self._get_linenums()
        instrs = self._as_instrs()

        ans = Text()
        for instr in instrs:
            if i := instr.isinst(RawInstr):
                ans += i.instr.format_raw(i, AsRawCtx(i, linenums)).text
            elif i := instr.isinst(Comment):
                ans += Text("# ", "ic10.comment")
                ans += i.instr.text
                for x in i.inputs_:
                    ans += " "
                    ans += format_val(x)
                ans += "\n"
            elif i := instr.isinst(EmitLabel):
                pass
            elif i := instr.isinst(EndPlaceholder):
                c = linenums.instr_lines[i]
                if c in linenums.label_lines.values():
                    ans += Text(f"# noop (jump here to exit)\n", "ic10.comment")
                    c += 1
                ans += Text(f"# {c} lines total (not including this line)", "ic10.comment")
            else:
                assert False

        return RawText(ans)

    def __rich__(self) -> RenderableType:
        if not self.did_regalloc:
            return self.frag.__rich__("Program")

        linenums = self._get_linenums()

        def gen():
            instrs = self._as_instrs()

            prev_cont = True
            for x in instrs:
                if not prev_cont:
                    yield ""
                prev_cont = x.continues
                yield x.__rich__()

        with FORMAT_SCOPE_CONTEXT.bind(self.frag.scope), FORMAT_LINENUMS.bind(linenums):
            return Group(*gen())
            # return Panel(
            #     Group(*gen()),
            #     title=Text("Program", "ic10.title"),
            #     title_align="left",
            # )
