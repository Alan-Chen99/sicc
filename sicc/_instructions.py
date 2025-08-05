from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING  # autoflake: skip
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Self
from typing import TypedDict
from typing import Unpack
from typing import cast
from typing import overload
from typing import override

from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.console import RenderableType
from rich.console import group
from rich.panel import Panel
from rich.text import Text

from ._core import FORMAT_ANNOTATE
from ._core import Block
from ._core import BoundInstr
from ._core import EffectBase
from ._core import EffectRes
from ._core import InstrBase
from ._core import InternalValLabel
from ._core import Label
from ._core import TypeList
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VarTS
from ._core import WriteMVar
from ._core import format_instr_list
from ._core import format_val
from ._core import get_type
from ._core import get_types
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._utils import cast_unchecked
from ._utils import mk_ordered_set
from ._utils import narrow_unchecked
from .config import verbose

# register_exclusion(__file__)


@dataclass(frozen=True)
class EffectExternal(EffectBase):
    # interacts with the outside world
    pass


################################################################################


class EmitLabel(InstrBase):
    in_types = (Label,)
    out_types = ()

    jumps = False

    @override
    def format(self, instr: BoundInstr[Self], /) -> Text:
        return format_val(instr.inputs_[0]) + ":"

    @override
    def writes(self, instr: BoundInstr[Self]) -> Iterable[EffectBase]:
        return []

    @override
    def defines_labels(self, instr: BoundInstr[Self], /) -> Iterable[Label]:
        (l,) = instr.inputs_
        assert not isinstance(l, Var)
        yield l


@dataclass(frozen=True)
class RawInstr(InstrBase):
    opcode: str
    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: TypeList[tuple[Var, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]

    continues: bool
    jumps: bool

    # def format_with_args_outputs(self, out_vars: tuple[Var, ...], *args: Value) -> Text:
    #     ans = Text()
    #     ans.append(self.opcode, "ic10.raw_opcode")
    #     if len(out_vars) > 0:
    #         ans += " ["
    #         ans += format_val(out_vars[0])
    #         for x in out_vars[1:]:
    #             ans += ", "
    #             ans += format_val(x)
    #         ans += "]"

    #     for x in args:
    #         ans += " "
    #         ans += format_val(x)

    #     return ans


class RawAsmOpts(TypedDict, total=False):
    continues: bool
    jumps: bool


@overload
def raw_asm[T: VarT](
    opcode: str, out_type: type[T], /, *args: Value, **kwargs: Unpack[RawAsmOpts]
) -> Var[T]: ...
@overload
def raw_asm(opcode: str, out_type: None, /, *args: Value, **kwargs: Unpack[RawAsmOpts]) -> None: ...


def raw_asm[T: VarT](
    opcode: str, out_type: type[T] | None, /, *args: Value, **kwargs: Unpack[RawAsmOpts]
) -> Var[T] | None:
    kwargs.setdefault("continues", True)
    kwargs.setdefault("jumps", True)

    if out_type is None:
        () = RawInstr(opcode, TypeList(get_types(*args)), TypeList(), **kwargs).emit(*args)
        return None
    else:
        (ans,) = RawInstr(
            opcode,
            TypeList(get_types(*args)),
            TypeList((out_type,)),
            **kwargs,
        ).emit(*args)
        return ans.check_type(out_type)


class AsmInstrBase(InstrBase):
    """instr that lowers directly to an asm op"""

    opcode: str
    in_types: VarTS
    out_types: VarTS

    @override
    def lower(self, *args: Value) -> tuple[Var, ...]:
        assert False
        # return RawInstr(
        #     self.opcode,
        #     TypeList(self.in_types),
        #     TypeList(self.out_types),
        #     continues=self.continues,
        #     jumps=self.jumps,
        # ).emit(*args)


################################################################################


class Move[T: VarT = Any](AsmInstrBase):
    jumps = False

    def __init__(self, typ: type[T]) -> None:
        self.opcode = "move"
        self.in_types = (typ,)
        self.out_types = (typ,)


class Stop(InstrBase):
    in_types = ()
    out_types = ()
    continues = False


class EndPlaceholder(InstrBase):
    in_types = ()
    out_types = ()
    continues = False


class Jump(AsmInstrBase):
    opcode = "j"
    in_types = (Label,)
    out_types = ()
    continues = False


class UnreachableChecked(InstrBase):
    in_types = ()
    out_types = ()
    continues = False

    @override
    def writes(self, instr: BoundInstr[Self]) -> None:
        return


class BlackBox[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, typ: type[T]) -> None:
        self.opcode = "move"
        self.in_types = (typ,)
        self.out_types = (typ,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectBase:
        return EffectExternal()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectBase:
        return EffectExternal()


################################################################################


class PredicateBase(AsmInstrBase):
    """
    child classes must be pure
    """

    out_types = (bool,)

    jumps = False

    def negate(self, instr: BoundInstr[Any]) -> BoundInstr[PredicateBase]:
        raise NotImplementedError()

    # def lower_neg(self, *args: Value) -> Var[bool]:
    #     (out_var,), bound = self.negate(*args)
    #     bound.emit()
    #     return out_var

    # def lower_cjump(self, *args: Value, label: InternalValLabel) -> None:
    #     assert self.opcode.startswith("s")
    #     raw_asm("b" + self.opcode.removeprefix("s"), None, *args, label)

    # def lower_neg_cjump(self, *args: Value, label: InternalValLabel) -> None:
    #     (_out_var,), bound = self.negate(*args)
    #     assert isinstance(bound.instr, PredicateBase)
    #     bound.instr.lower_cjump(*bound.inputs, label=label)


class PredVar(PredicateBase):
    """a predicate that is just a Bool"""

    opcode = "move"
    in_types = (bool,)

    @override
    def negate(self, instr: BoundInstr[Self]):
        return Not().bind(instr.outputs_, *instr.inputs_)

    # @override
    # def lower_cjump(
    #     self, a: InteralBool, label: InternalValLabel
    # ) -> None:
    #     from ._instructions import PredLE

    #     PredLE().lower_cjump(1, a, label=label)


class Not(PredicateBase):
    opcode = "not"
    in_types = (bool,)

    @override
    def negate(self, instr: BoundInstr[Self]):
        return PredVar().bind(instr.outputs_, *instr.inputs_)

    # @override
    # def lower_cjump(
    #     self, a: InteralBool, label: InternalValLabel
    # ) -> None:
    #     from ._instructions import PredLT

    #     PredLT().lower_cjump(a, 1, label=label)


class Branch(InstrBase):
    in_types = (bool, Label, Label)
    out_types = ()
    continues = False


class CondJump(InstrBase):
    """
    jumps to a label if pred evaluates equal to "jump_on"; otherwise continue

    tracing does not emit this; only used after optimize when concating blocks
    """

    def __init__(self):
        self.in_types = (bool, Label)
        self.out_types = ()
        self.continues = True


################################################################################
# MATH
################################################################################


class AddF(AsmInstrBase):
    opcode = "add"
    in_types = (float, float)
    out_types = (float,)


class AddI(AsmInstrBase):
    opcode = "add"
    in_types = (int, int)
    out_types = (int,)


################################################################################
# PREDICATES
################################################################################


class PredLT(PredicateBase):
    opcode = "slt"
    in_types = (float, float)

    @override
    def negate(self, instr: BoundInstr[Self]):
        a, b = instr.inputs_
        return PredLE().bind(instr.outputs_, b, a)


class PredLE(PredicateBase):
    opcode = "sle"
    in_types = (float, float)

    @override
    def negate(self, instr: BoundInstr[Self]):
        a, b = instr.inputs_
        return PredLT().bind(instr.outputs_, b, a)


@dataclass
class Bundle[*Ts = * tuple[BoundInstr[Any], ...]](InstrBase):
    # False, i -> var is inputs[i]
    # True, i -> var is outputs[i]
    var_info: list[tuple[bool, int]]

    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: TypeList[tuple[Var, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]

    children: list[tuple[InstrBase, tuple[int, ...], tuple[int, ...], DebugInfo]]

    # # def bundles(self, instr: BoundInstr[Any], /) -> Iterable[BoundInstr] | None:
    # #     """
    # #     If not None, instr is a bundle of several instructions.

    # #     Every time its called, it should
    # #     (1) return a list of instructions with a new/different instr id.
    # #           this list must not contain [self]
    # #     (2) unlike BoundInstr objects, internal vars in the bundle must be in the
    # #           "output" field of a BoundInstr

    # #     this features is added later; some code might not be aware of this and may make mistakes.
    # #     TODO:
    # #     (1) check EmitLabel. now this is no longer the only thing that can produce a label
    # #     (2) i think some previous code incorrectly assmed instrs have at most one output
    # #     """
    # #     return None

    def parts(self, instr: BoundInstr[Self]) -> tuple[*Ts]:
        vars = [instr.outputs[x] if is_out else instr.inputs[x] for is_out, x in self.var_info]
        ans: list[BoundInstr] = []
        for i, (child, args, outs, dbg) in enumerate(self.children):
            arg_vals = tuple(vars[j] for j in args)
            out_vars = tuple(cast(Var, vars[j]) for j in outs)
            ans.append(BoundInstr((*instr.id, i), child, arg_vals, out_vars, dbg))

        return cast_unchecked(tuple(ans))

    @classmethod
    def from_parts(cls, *instrs: *Ts) -> BoundInstr[Self]:
        if TYPE_CHECKING:
            assert narrow_unchecked(instrs, tuple[BoundInstr, ...])

        vars: OrderedSet[Value] = mk_ordered_set()
        out_vars: set[Var] = set()

        children: list[tuple[InstrBase, tuple[int, ...], tuple[int, ...], DebugInfo]] = []

        for instr in instrs:
            arg_idxs = tuple(vars.add(x) for x in instr.inputs)
            out_idxs = tuple(vars.add(x) for x in instr.outputs)
            out_vars |= set(instr.outputs)
            children.append((instr.instr, arg_idxs, out_idxs, instr.debug))

        var_info: list[tuple[bool, int]] = []
        inputs: list[Value] = []
        outputs: list[Var] = []

        for x in vars:
            if isinstance(x, Var) and x in out_vars:
                var_info.append((True, len(outputs)))
                outputs.append(x)
            else:
                var_info.append((False, len(inputs)))
                inputs.append(x)

        ans = cls(
            var_info=var_info,
            in_types=TypeList((get_type(x) for x in inputs)),
            out_types=TypeList((x.type for x in outputs)),
            children=children,
        )

        return ans.bind_untyped(tuple(outputs), *inputs)

    @staticmethod
    def from_block(b: Block) -> BoundInstr[Bundle]:
        assert b.end.isinst(EndPlaceholder)
        with add_debug_info(b.debug):
            return Bundle.from_parts(*b.contents[:-1])

    @override
    def get_continues(self, instr: BoundInstr[Self]) -> bool:
        return instr.unpack_untyped()[-1].continues

    @override
    def jumps_to(self, instr: BoundInstr[Self]) -> Iterable[InternalValLabel]:
        for part in instr.unpack_untyped():
            yield from part.jumps_to()

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        for part in instr.unpack_untyped():
            yield from part.reads()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        for part in instr.unpack_untyped():
            yield from part.writes()

    @override
    def defines_labels(self, instr: BoundInstr[Self]) -> Iterable[Label]:
        for part in instr.unpack_untyped():
            yield from part.defines_labels()

    @override
    def format_with_anno(self, instr: BoundInstr[Self], /) -> RenderableType:
        parts = instr.unpack_untyped()

        comment = Text()
        if loc_info := instr.debug.location_info_brief():
            comment.append("  # " + loc_info, "ic10.comment")

        if annotation := FORMAT_ANNOTATE.value(instr):
            comment.append("  # ", "ic10.comment")
            comment.append(annotation)

        @group()
        def mk_group() -> Iterator[RenderableType]:
            if len(comment) > 0:
                yield comment[2:]
            yield from format_instr_list(list(parts))

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


@dataclass
class PredBranch(
    Bundle[
        BoundInstr[PredicateBase],
        BoundInstr[Branch],
    ]
):
    pass


class JumpAndLink(
    Bundle[
        BoundInstr[WriteMVar],
        BoundInstr[Jump],
        BoundInstr[EmitLabel],
    ]
):
    pass


class CondJumpAndLink(
    Bundle[
        BoundInstr[WriteMVar],
        BoundInstr[PredicateBase],
        BoundInstr[CondJump],
        BoundInstr[EmitLabel],
    ]
):
    pass
