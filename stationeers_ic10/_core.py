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
from rich.panel import Panel
from rich.text import Text

from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import debug_info
from ._utils import ByIdMixin
from ._utils import Cell
from ._utils import cast_unchecked
from ._utils import disjoint_union
from ._utils import get_id
from ._utils import narrow_unchecked
from ._utils import safe_cast

if TYPE_CHECKING:
    from ._api import UserValue
    from ._api import Variable
    from ._instructions import EmitLabel


# register_exclusion(__file__)

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


def _default_format(self: InstrBase, *args: Value) -> Text:
    ans = Text()
    mark = self.jumps and any(get_type(x) == Label for x in args)
    mark |= not self.continues
    ans.append(type(self).__name__, "ic10.jump" if mark else "ic10.opcode")
    for x in args:
        ans += " "
        ans += format_val(x)
    return ans


LowerRes = tuple[Var, ...] | Var | None


class InstrBase(abc.ABC):
    # required overrides
    in_types: VarTS
    out_types: VarTS
    lower: Callable[..., LowerRes] = _default_lower

    # optional overrides

    #: continues to the next instruction
    continues: bool = True

    #: if True, may jump to any label that is a input
    #: if False, never jumps
    jumps: bool = True

    # if overriding this, "jumps" variable have no effect
    def jumps_to(self, instr: BoundInstr[Any]) -> Iterable[InternalValLabel]:
        if self.jumps:
            return [x for x in instr.inputs if can_cast_val(x, Label)]
        return []

    format_with_args: Callable[..., Text] = _default_format

    # impure operations MUST override reads and/or writes
    def reads(self, instr: BoundInstr[Any], /) -> EffectRes:
        return []

    def writes(self, instr: BoundInstr[Any], /) -> EffectRes:
        if len(self.out_types) == 0 and len(instr.jumps_to()) == 0:
            raise NotImplementedError(f"{type(self)} returns nothing, so it must override 'writes'")
        return []

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
        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)
            assert narrow_unchecked(args, tuple[Value, ...])
            assert narrow_unchecked(out_vars, tuple[Var, ...])

        self.check_inputs(*args)
        self.check_outputs(*out_vars)
        return BoundInstr(get_id(), self, args, out_vars, debug_info())

    def create_bind[*I, O](
        self: InstrTypedWithArgs[tuple[*I], O], *args: *I
    ) -> tuple[O, BoundInstr]:
        from ._tracing import mk_var

        if TYPE_CHECKING:
            assert isinstance(self, InstrBase)
            assert narrow_unchecked(args, tuple[Value, ...])

        self.check_inputs(*args)
        out_vars = tuple(mk_var(x) for x in self.out_types)

        ans = BoundInstr(get_id(), self, args, out_vars, debug_info())
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
    id: int
    instr: B_co
    inputs: tuple[Value, ...]
    outputs: tuple[Var, ...]
    debug: DebugInfo

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

    def jumps_to(self: BoundInstr) -> list[InternalValLabel]:
        return list(self.instr.jumps_to(self))

    def __rich__(self) -> Text:
        from ._instructions import RawInstr

        assert narrow_unchecked(self, BoundInstr)

        if me := self.isinst(RawInstr):
            return me.instr.format_with_args_outputs(self.outputs, *self.inputs)

        ans = Text()
        if len(self.outputs) > 0:
            ans += format_val(self.outputs[0])
            for x in self.outputs[1:]:
                ans += ", "
                ans += format_val(x)
            ans += " = "
        ans += self.instr.format_with_args(*self.inputs)

        loc_info = self.debug.location_info()
        if loc_info:
            ans.append("  # " + loc_info, "ic10.comment")

        annotation = FORMAT_ANNOTATE.value(self)
        if annotation:
            ans.append("  # ", "ic10.comment")
            ans.append(annotation)

        return ans

    def __repr__(self):
        return repr(self.__rich__().plain)

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

        return BoundInstr(get_id(), self.instr, new_inputs, new_outputs, debug_info())

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
    def format_with_args(self) -> Text:
        return self.s._format()

    @override
    def reads(self, instr: BoundInstr[Any], /) -> EffectBase:
        return EffectMvar(self.s)


class WriteMVar[T: VarT = Any](InstrBase):
    s: MVar[T]

    jumps = False

    def __init__(self, s: MVar[T]) -> None:
        self.s = s
        self.in_types = (s.type,)
        self.out_types = ()

    @override
    def format_with_args(self, val: Value[T]) -> Text:
        return self.s._format() + " = " + format_val(val)

    @override
    def writes(self, instr: BoundInstr[Any], /) -> EffectBase:
        return EffectMvar(self.s)


################################################################################

MapInstrsRes = BoundInstr | Iterable[BoundInstr] | None
MapInstrsFn = Callable[[BoundInstr], MapInstrsRes]


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
        assert not ans.instr.continues
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
        from ._instructions import EmitLabel

        def content() -> Iterator[RenderableType]:
            prev_is_label = True
            prev_cont = True
            for x in self.contents:
                sep = not prev_cont
                if x.isinst(EmitLabel):
                    if not prev_is_label:
                        sep = True
                    prev_is_label = True
                else:
                    prev_is_label = False
                prev_cont = x.instr.continues

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

    def map_instrs(self, fn: MapInstrsFn) -> None:
        def get(x: BoundInstr) -> list[BoundInstr]:
            with add_debug_info(x.debug):
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

    def all_instrs(self) -> list[BoundInstr]:
        return [x for b in self.blocks.values() for x in b.contents]

    def map_instrs(self, fn: MapInstrsFn) -> None:
        ans: dict[Label, Block] = {}
        for x in self.blocks.values():
            x.map_instrs(fn)
            assert x.label not in ans
            ans[x.label] = x
        self.blocks = ans
        self.basic_check()

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
