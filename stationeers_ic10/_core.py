from __future__ import annotations

import abc
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Final
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Never
from typing import Protocol
from typing import Self
from typing import TypeGuard
from typing import TypeVar
from typing import overload
from typing import override
from typing import runtime_checkable

from ordered_set import OrderedSet
from rich.console import Group
from rich.console import RenderableType
from rich.console import group
from rich.panel import Panel
from rich.text import Text

from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import clear_debug_info
from ._diagnostic import debug_info
from ._diagnostic import register_exclusion
from ._utils import ByIdMixin
from ._utils import Cell
from ._utils import cast_unchecked
from ._utils import disjoint_union
from ._utils import get_id
from ._utils import narrow_unchecked
from ._utils import safe_cast
from .config import verbose

if TYPE_CHECKING:
    from ._api import UserValue
    from ._api import Variable
    from ._instructions import EmitLabel


register_exclusion(__file__)

counter = Cell(0)


@runtime_checkable
class AsLiteral(Protocol):
    def as_literal(self) -> VarT: ...


type VarTS = tuple[type[VarT], ...]


T_co = TypeVar("T_co", covariant=True, bound="VarT", default="VarT")


@dataclass(frozen=True, eq=False)
class Var(Generic[T_co], ByIdMixin):
    """
    internal varaible used by the compiler.
    unlike Variable, equality and comparison is by variable id.

    A Var must always be assigned in one place. A MVar may be assigned any number of times.
    """

    type: type[T_co]
    id: int
    #: internal check. user invalid uses should be caught when the Var is found not in _CUR_SCOPE
    live: Cell[bool]
    debug: DebugInfo

    def check_type[T: VarT](self, typ: type[T]) -> Var[T]:
        if not can_cast_implicit(self.type, typ):
            raise TypeError(f"not possible to use {self.type} as {typ}")
        return cast_unchecked(self)

    def __repr__(self) -> str:
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


VarT = bool | int | float | str | Label | AsLiteral

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
    if t1 == t2:
        return True
    if (t1, t2) == (int, float):
        return True
    if (t1, t2) == (bool, int):
        return True
    if (t1, t2) == (bool, float):
        return True
    return False


def can_cast_implicit_many(t1: VarTS, t2: VarTS) -> bool:
    if len(t1) != len(t2):
        return False
    for x, y in zip(t1, t2):
        if not can_cast_implicit(x, y):
            return False
    return True


def can_cast_val[T: VarT](v: Value, typ: type[T]) -> TypeGuard[Value[T]]:
    return can_cast_implicit(get_type(v), typ)


class TypeList[T](tuple[type[VarT], ...]):
    def _inv_marker(self, x: T) -> T: ...


class _InstrTypedIn[I](Protocol):
    in_types: Final[I]


class _InstrTypedOut[O](Protocol):
    out_types: Final[O]


class InstrTypedWithArgs[I, O](Protocol):
    def _static_in_typing_helper(self, x: I, /) -> None: ...
    def _static_out_typing_helper(self) -> O: ...


class InstrTypedWithArgs_api[I, O](Protocol):
    def _static_in_typing_helper_api(self, x: I, /) -> None: ...
    def _static_out_typing_helper_api(self) -> O: ...


def _default_lower(self: InstrBase, *args: Value) -> Never:
    raise NotImplementedError(f"not implemented, or {type(self)} is not supposed to be lowered")


def _default_annotate(x: BoundInstr) -> str:
    return ""


FORMAT_SCOPE_CONTEXT: Cell[Scope] = Cell()
FORMAT_ANNOTATE: Cell[Callable[[BoundInstr], str | Text]] = Cell(_default_annotate)


def get_style(typ: type[VarT]) -> str:
    if issubclass(typ, (bool, int, float, str)):
        return "ic10." + typ.__name__
    if issubclass(typ, Label):
        return "ic10.label"
    return "ic10.other"


def format_val(v: Value) -> Text:
    typ = get_type(v)
    if issubclass(typ, Label):
        priv = (f := FORMAT_SCOPE_CONTEXT.get()) and (v in f.private_labels)
        return Text(repr(v), "ic10.label_private" if priv else "ic10.label")

    return Text(repr(v), get_style(typ))


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


class BundlesProto[T](Protocol):
    def bundles(self, instr: BoundInstr[Any], /) -> T: ...


class InstrBase(abc.ABC):
    # required overrides
    in_types: VarTS
    out_types: VarTS
    lower: Callable[..., LowerRes] = _default_lower

    # optional overrides

    def bundles(self, instr: BoundInstr[Any], /) -> Iterable[BoundInstr] | None:
        """
        If not None, instr is a bundle of several instructions.

        Every time its called, it should
        (1) return a list of instructions with a new/different instr id.
              this list must not contain [self]
        (2) unlike BoundInstr objects, internal vars in the bundle must be in the
              "output" field of a BoundInstr

        this features is added later; some code might not be aware of this and may make mistakes.
        TODO:
        (1) check EmitLabel. now this is no longer the only thing that can produce a label
        (2) i think some previous code incorrectly assmed instrs have at most one output
        """
        return None

    #: continues to the next instruction
    continues: bool = True

    def get_continues(self, instr: BoundInstr[Any], /) -> bool:
        """if overriding this, "continues" variable have no effect"""
        if (unpacked := instr.unpack_rec()) is not None:
            return unpacked[-1].continues
        return self.continues

    #: if True, may jump to any label that is a input
    #: if False, never jumps
    jumps: bool = True

    def jumps_to(self, instr: BoundInstr[Any], /) -> Iterable[InternalValLabel]:
        """if overriding this, "jumps" variable have no effect"""
        if (unpacked := instr.unpack_rec()) is not None:
            for part in unpacked:
                yield from part.jumps_to()
            return

        if self.jumps:
            for x in instr.inputs:
                if can_cast_val(x, Label):
                    yield x

    # impure operations MUST override reads and/or writes
    # writes does NOT imply reads
    def reads(self, instr: BoundInstr[Any], /) -> EffectRes:
        if (unpacked := instr.unpack_rec()) is not None:
            for part in unpacked:
                yield from part.reads()
            return

    def writes(self, instr: BoundInstr[Any], /) -> EffectRes:
        if (unpacked := instr.unpack_rec()) is not None:
            for part in unpacked:
                yield from part.writes()
            return

        if len(self.out_types) == 0 and len(instr.jumps_to()) == 0 and instr.continues:
            raise NotImplementedError(f"{type(self)} returns nothing, so it must override 'writes'")

    def defines_labels(self, instr: BoundInstr[Any], /) -> Iterable[Label]:
        """does it contain EmitLabel or equirvalent?"""
        if (unpacked := instr.unpack_rec()) is not None:
            for part in unpacked:
                yield from part.defines_labels()
        return

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

    def format_with_anno(self, instr: BoundInstr[Any], /) -> RenderableType:
        comment = Text()
        if loc_info := instr.debug.location_info_brief():
            comment.append("  # " + loc_info, "ic10.comment")

        if annotation := FORMAT_ANNOTATE.value(instr):
            comment.append("  # ", "ic10.comment")
            comment.append(annotation)

        if (parts := instr.unpack()) is not None:

            @group()
            def mk_group() -> Iterator[RenderableType]:
                if len(comment) > 0:
                    yield comment[2:]
                yield from format_instr_list(parts)

            if verbose.value >= 2:
                title = self.format(instr)
            else:
                title = Text(type(self).__name__, "ic10.title")

            return Panel(
                mk_group(),
                title=title,
                # title=self.format(instr),
                title_align="left",
            )

        ans = Text()
        ans += self.format(instr)
        ans += comment

        return ans

    # format_with_args: Callable[..., Text] = _default_format

    ################################################################################
    # stub methods for static typing
    ################################################################################

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
    ) -> Variable[A]: ...
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
    def _static_in_typing_helper_api(self: Any, x: Any, /) -> None: ...

    ################################################################################

    def check_inputs(self, *args: Value) -> None:
        for x in args:
            _ck_val(x)
        in_types = self.in_types
        arg_types = get_types(*args)
        if not can_cast_implicit_many(arg_types, in_types):
            raise TypeError(f"not possible to use {arg_types} as {in_types}")

    def check_outputs(self, *args: Var) -> None:
        for x in args:
            _ck_val(x)
        out_types = self.out_types
        arg_types = get_types(*args)
        if not can_cast_implicit_many(out_types, arg_types):
            raise TypeError(f"not possible to use {out_types} as {arg_types}")

    def bind[*I, O](
        self: InstrTypedWithArgs[tuple[*I], O], out_vars: O, /, *args: *I
    ) -> BoundInstr:
        return self.bind_untyped(out_vars, *args)  # pyright: ignore

    def bind_untyped[*I, O](self, out_vars: tuple[Var, ...], /, *args: Value) -> BoundInstr[Self]:
        self.check_inputs(*args)
        self.check_outputs(*out_vars)
        return BoundInstr((get_id(),), self, args, out_vars, debug_info())

    def create_bind[*I, O](
        self: InstrTypedWithArgs[tuple[*I], O], *args: *I
    ) -> tuple[O, BoundInstr]:
        from ._tracing import mk_var

        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)
            assert narrow_unchecked(args, tuple[Value, ...])

        self.check_inputs(*args)
        out_vars = tuple(mk_var(x) for x in self.out_types)

        ans = BoundInstr((get_id(),), self, args, out_vars, debug_info())
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
    def inputs_[I1, O1](self: BoundInstr[InstrTypedWithArgs[I1, O1]]) -> I1:
        return cast_unchecked(self.inputs)

    @property
    def outputs_[I1, O1](self: BoundInstr[InstrTypedWithArgs[I1, O1]]) -> O1:
        return cast_unchecked(self.outputs)

    def check_scope(self):
        for x in self.inputs:
            _ck_val(x)
        for x in self.outputs:
            _ck_val(x)

    @overload
    def unpack_typed[*Ts](self: BoundInstr[BundlesProto[tuple[*Ts]]]) -> tuple[*Ts]: ...
    @overload
    def unpack_typed[T](
        self: BoundInstr[BundlesProto[Iterable[BoundInstr[T]]]],
    ) -> tuple[BoundInstr[T], ...]: ...

    def unpack_typed(self):
        ans = cast_unchecked(self.unpack())  # pyright: ignore
        assert ans is not None
        return ans

    def unpack(self: BoundInstr) -> list[BoundInstr] | None:
        parts = self.instr.bundles(self)
        if parts is None:
            return None
        parts = list(parts)
        return [
            BoundInstr((*self.id, i), x.instr, x.inputs, x.outputs, x.debug)
            for i, x in enumerate(parts)
        ]

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

    def isinst[T: InstrBase](self, instr_type: type[T]) -> BoundInstr[T] | None:
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

        return BoundInstr((get_id(),), self.instr, new_inputs, new_outputs, debug_info())

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


################################################################################


@dataclass(frozen=True, eq=False)
class MVar[T: VarT = Any](ByIdMixin):
    type: type[T]
    id: int
    debug: DebugInfo

    def read(self) -> Var[T]:
        return ReadMVar(self).call()

    def write(self, v: Value[T]) -> None:
        () = WriteMVar(self).emit(v)

    def __repr__(self) -> str:
        return f"%s{self.id}"

    def _format(self):
        priv = (f := FORMAT_SCOPE_CONTEXT.get()) and (self in f.private_mvars)
        ans = Text("", "underline" if priv else "underline reverse")
        ans.append(repr(self), get_style(self.type))
        return Text() + ans


@dataclass(frozen=True)
class EffectMvar(EffectBase):
    s: MVar

    def known_distinct(self, other: Self) -> bool:
        return self.s != other.s


class SubMVar:
    pass


class ReadMVar[T: VarT = Any](InstrBase):
    s: MVar[T]

    jumps = False

    def __init__(self, s: MVar[T]) -> None:
        self.s = s
        self.in_types = ()
        self.out_types = (s.type,)

    @override
    def format_expr_part(self, instr: BoundInstr[Self], /) -> Text:
        return self.s._format()

    @override
    def reads(self, instr: BoundInstr[Self], /) -> EffectBase:
        return EffectMvar(self.s)


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

    def get_vars(self) -> None:
        pass


################################################################################
