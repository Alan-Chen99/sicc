from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Final, Generic, Iterator, Never, Protocol, TypeGuard, TypeVar, overload

from ordered_set import OrderedSet
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from ._diagnostic import DebugInfo, debug_info
from ._utils import Cell, cast_unchecked, disjoint_union, late_fn, narrow_unchecked

counter = Cell(0)


def get_id() -> int:
    ans = counter.value
    counter.value = ans + 1
    return ans


class AsLiteral(Protocol):
    def as_literal(self) -> VarT: ...


type VarT = bool | int | float | str | Label | AsLiteral
type VarTS = tuple[type[VarT], ...]


T_co = TypeVar("T_co", covariant=True, bound=VarT, default=VarT)


@dataclass(frozen=True)
class Var(Generic[T_co]):
    type: type[T_co]
    id: int
    debug: DebugInfo

    def check_type[T: VarT](self, typ: type[T]) -> Var[T]:
        if not can_cast_implicit(self.type, typ):
            raise TypeError(f"not possible to use {self.type} as {typ}")
        return cast_unchecked(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and other.id == self.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"%v{self.id}"

    ##########

    __add__ = late_fn(lambda: add)

    def __lt__(self: Float, other: Float) -> Var[bool]:
        return PredLT().call(self, other)

    def __gt__(self: Float, other: Float) -> Var[bool]:
        return PredLT().call(other, self)

    def __le__(self: Float, other: Float) -> Var[bool]:
        return PredLE().call(self, other)

    def __ge__(self: Float, other: Float) -> Var[bool]:
        return PredLE().call(other, self)

    ##########


def mk_var[T: VarT](typ: type[T]) -> Var[T]:
    return Var(typ, get_id(), debug_info())


@dataclass(frozen=True)
class Label:
    name: str
    debug: DebugInfo

    #: Non-implicit labels are generally user created
    #: they are always kept
    implicit: bool

    def __eq__(self, other: object):
        return isinstance(other, Label) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def _mark_private_here(self) -> None:
        """mark self as private in the current tracing fragment"""
        mark_label_private(self)


LabelLike = Label | str


def mk_label(l: LabelLike | None = None, *, implicit: bool = False) -> Label:
    if isinstance(l, Label):
        return l
    if l is None:
        if implicit:
            l = f"_implicit_{get_id()}"
        else:
            l = f"anon_{get_id()}"
    return Label(l, debug_info(), implicit=implicit)


def mk_internal_label(prefix: str, id: int | None = None) -> Label:
    if id is None:
        id = get_id()
    ans = mk_label(f"_{prefix}_{id}", implicit=True)
    ans._mark_private_here()
    return ans


type Value[T: VarT = VarT] = Var[T] | T

Bool = Value[bool]
Int = Value[int]
Float = Value[float]
ValLabel = Value[Label]


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


def _default_lower(self: InstrBase, *args: Value) -> Never:
    raise NotImplementedError(f"not implemented, or {type(self)} is not supposed to be lowered")


FORMAT_CTX: Cell[Fragment] = Cell()


def format_val(v: Value) -> Text:
    typ = get_type(v)
    if typ in [bool, int, float, str]:
        return Text(repr(v), "ic10." + typ.__name__)

    if issubclass(typ, Label):
        priv = (f := FORMAT_CTX.get()) and (v in f.private_labels)
        return Text(repr(v), "ic10.label_private" if priv else "ic10.label")

    return Text(repr(v), "ic10.other")


def _default_format(self: InstrBase, *args: Value) -> Text:
    ans = Text()
    mark = self.jumps and any(get_type(x) == Label for x in args)
    mark |= not self.continues
    ans.append(type(self).__name__, "ic10.jump" if mark else "ic10.opcode")
    for x in args:
        ans += " "
        ans += format_val(x)
    return ans


class InstrBase(abc.ABC):
    # required overrides
    in_types: VarTS
    out_types: VarTS
    lower: Callable[..., Any] = _default_lower

    # optional overrides

    #: continues to the next instruction
    continues: bool = True

    #: if True, may jump to any label that is a input
    #: if False, never jumps
    jumps: bool = True

    format_with_args: Callable[..., Text] = _default_format

    ################################################################################
    # stub methods for static typing
    ################################################################################

    def _marker[T](self, x: T, /) -> T: ...

    @overload
    def _static_out_typing_helper(self: _InstrTypedOut[tuple[()]]) -> tuple[()]: ...
    @overload
    def _static_out_typing_helper[A: VarT](self: _InstrTypedOut[tuple[type[A]]]) -> tuple[Var[A]]: ...
    @overload
    def _static_out_typing_helper[T](self: _InstrTypedOut[TypeList[T]]) -> T: ...
    def _static_out_typing_helper(self: Any) -> Any: ...

    @overload
    def _static_in_typing_helper(self: _InstrTypedIn[tuple[()]], x: tuple[()], /) -> None: ...
    @overload
    def _static_in_typing_helper[A: VarT](self: _InstrTypedIn[tuple[type[A]]], x: tuple[Value[A]], /) -> None: ...
    @overload
    def _static_in_typing_helper[A: VarT, B: VarT](
        self: _InstrTypedIn[tuple[type[A], type[B]]], x: tuple[Value[A], Value[B]], /
    ) -> None: ...
    @overload
    def _static_in_typing_helper[T](self: _InstrTypedIn[TypeList[T]], x: T, /) -> None: ...
    def _static_in_typing_helper(self: Any, x: Any, /) -> None: ...

    ################################################################################

    # def get_pure(self) -> bool:
    #     return False

    # def get_side_effect_free(self) -> bool:
    #     return False

    def check_inputs(self, *args: Value):
        in_types = self.in_types
        arg_types = get_types(*args)
        if not can_cast_implicit_many(arg_types, in_types):
            raise TypeError(f"not possible to use {arg_types} as {in_types}")

    def check_outputs(self, *args: Var):
        out_types = self.out_types
        arg_types = get_types(*args)
        if not can_cast_implicit_many(out_types, arg_types):
            raise TypeError(f"not possible to use {out_types} as {arg_types}")

    def bind[*I, O](self: InstrTypedWithArgs[tuple[*I], O], out_vars: O, /, *args: *I) -> BoundInstr[tuple[*I], O]:
        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)

        self.check_inputs(*cast_unchecked(args))
        self.check_outputs(*cast_unchecked(out_vars))

        return BoundInstr(self, args, out_vars, debug_info())

    def create_bind[*I, O](self: InstrTypedWithArgs[tuple[*I], O], *args: *I) -> tuple[O, BoundInstr[tuple[*I], O]]:
        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)

        self.check_inputs(*cast_unchecked(args))
        out_vars: O = cast_unchecked(tuple(mk_var(x) for x in self.out_types))

        return out_vars, BoundInstr(self, args, out_vars, debug_info())

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


@dataclass(frozen=True)
class BoundInstr[I = tuple[Value, ...], O = tuple[Var, ...]]:
    instr: InstrTypedWithArgs[I, O]
    inputs: I
    ouputs: O
    debug: DebugInfo

    def __rich__(self) -> RenderableType:
        assert narrow_unchecked(self, BoundInstr)

        ans = Text()
        if len(self.ouputs) > 0:
            ans += format_val(self.ouputs[0])
            for x in self.ouputs[1:]:
                ans += ", "
                ans += format_val(x)
            ans += " = "
        ans += self.instr_.format_with_args(*self.inputs)
        return ans

    @staticmethod
    def isinst[I1, O1](
        me: BoundInstr[Any, Any], instr_type: type[InstrTypedWithArgs[I1, O1]]
    ) -> TypeGuard[BoundInstr[I1, O1]]:
        return isinstance(me.instr, instr_type)

    def check[I1, O1](self, instr_type: type[InstrTypedWithArgs[I1, O1]]) -> BoundInstr[I1, O1]:
        assert BoundInstr.isinst(self, instr_type)
        return self

    @property
    def instr_(self) -> InstrBase:
        assert isinstance(self.instr, InstrBase)
        return self.instr

    def emit(self) -> None:
        emit_bound(self)


################################################################################


@dataclass(frozen=True)
class MVar[T: VarT = Any]:
    type: type[T]
    id: int
    debug: DebugInfo

    @property
    def value(self) -> Var[T]:
        return ReadMVar(self).call()

    @value.setter
    def value(self, v: Var[T]) -> None:
        () = WriteMVar(self).emit(v)


class ReadMVar[T: VarT](InstrBase):
    s: MVar[T]

    jumps = False

    def __init__(self, s: MVar[T]) -> None:
        self.s = s
        self.in_types = ()
        self.out_types = (s.type,)


class WriteMVar[T: VarT](InstrBase):
    s: MVar[T]

    jumps = False

    def __init__(self, s: MVar[T]) -> None:
        self.s = s
        self.in_types = (s.type,)
        self.out_types = ()


################################################################################


@dataclass
class Block:
    """
    Must start with a EmitLabel and end with something with continues=False
    """

    contents: list[BoundInstr]
    debug: DebugInfo

    @property
    def label(self) -> Label:

        (ans,) = self.contents[0].check(EmitLabel).inputs
        # EmitLabel must be a constant Label
        assert isinstance(ans, Label)
        return ans

    @property
    def end(self) -> BoundInstr:
        ans = self.contents[-1]
        assert not ans.instr_.continues
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
        def content() -> Iterator[RenderableType]:
            prev_is_label = True
            prev_cont = True
            for x in self.contents:
                sep = not prev_cont
                if BoundInstr.isinst(x, EmitLabel):
                    if not prev_is_label:
                        sep = True
                    prev_is_label = True
                else:
                    prev_is_label = False
                prev_cont = x.instr_.continues

                if sep:
                    yield ""
                # eagerly, so that FORMAT_CTX takes effect
                yield x.__rich__()

        return Panel(
            Group(*content()),
            # title=format_val(self.label),
            # title = "Block",
            # title_align="left",
        )


@dataclass
class Fragment:
    """
    a set of blocks.

    there is no default entry or exit;
    the fragment is enterred if jumped to a tag in it, and exitted if jumpped out.

    all Var is private and must not be outside.
    MVar that is not in private_mvars may be used outside.
    """

    #: declares the fragment to have completed tracing
    #: it should no longer be modified afterwards
    finished_init: bool

    #: body of the fragment
    blocks: dict[Label, Block]

    #: private to this fragment
    private_mvars: OrderedSet[MVar]

    #: Labels that must not be explicity referenced outside this Fragment
    #:
    #: they may still be assigned to variables and used outside;
    #: whether that happens is inferred in compute_label_provenance
    #:
    private_labels: OrderedSet[Label]

    def __rich__(self) -> RenderableType:
        def content() -> Iterator[RenderableType]:
            if not self.finished_init:
                yield "(unfinished)"
            for b in self.blocks.values():
                # eagerly, so that FORMAT_CTX takes effect
                yield b.__rich__()

        with FORMAT_CTX.bind(self):
            return Panel(
                Group(*content()),
                title=Text(type(self).__name__, "ic10.title"),
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

        self.private_mvars = disjoint_union(self.private_mvars, child.private_mvars)

        self.private_labels = disjoint_union(self.private_labels, child.private_labels)


################################################################################

from ._instructions import EmitLabel, PredLE, PredLT
from ._tracing import emit_bound, mark_label_private
from .functions import add
