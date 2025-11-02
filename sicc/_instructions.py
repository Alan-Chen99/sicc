from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Never
from typing import Self
from typing import Sequence
from typing import cast
from typing import final
from typing import override

from rich import print as print  # autoflake: skip
from rich.console import RenderableType
from rich.console import group
from rich.panel import Panel
from rich.text import Text

from ._core import AsRawCtx
from ._core import Block
from ._core import BoundInstr
from ._core import ConstEval
from ._core import EffectBase
from ._core import EffectMvar
from ._core import EffectRes
from ._core import InstrBase
from ._core import InteralBool
from ._core import InternalValLabel
from ._core import Label
from ._core import MVar
from ._core import PinType
from ._core import RawText
from ._core import RegallocPref
from ._core import RegallocSkip
from ._core import RegallocTie
from ._core import Register
from ._core import TypeList
from ._core import Undef
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VarTS
from ._core import VirtualConst
from ._core import WriteMVar
from ._core import db_internal
from ._core import format_instr_list
from ._core import format_raw_val
from ._core import format_val
from ._core import format_vals
from ._core import get_type
from ._core import get_types
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import clear_debug_info
from ._diagnostic import track_caller
from ._utils import ByIdMixin
from ._utils import cast_unchecked
from ._utils import narrow_unchecked
from .config import verbose

# register_exclusion(__file__)


@dataclass(frozen=True)
class EffectExternal(EffectBase):
    # interacts with the outside world
    pass


################################################################################


@dataclass
class EmitLabel(InstrBase):
    in_types = (Label,)
    out_types = ()

    jumps = False

    allow_split: bool = True

    @override
    def format(self, instr: BoundInstr[Self], /) -> Text:
        return format_val(instr.inputs_[0], Label) + ":"

    @override
    def writes(self, instr: BoundInstr[Self]) -> Iterable[EffectBase]:
        return []

    @override
    def defines_labels(self, instr: BoundInstr[Self], /) -> Iterable[Label]:
        (l,) = instr.inputs_
        assert not isinstance(l, Var | VirtualConst)
        yield l


@dataclass(frozen=True)
class RawInstr(InstrBase):
    opcode: str
    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: TypeList[tuple[Var, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]

    _reads: list[EffectBase]
    _writes: list[EffectBase]
    continues: bool
    jumps: bool

    src_instr: type[InstrBase] | None = None

    @override
    def reads(self, instr: BoundInstr[Self]):
        return self._reads

    @override
    def writes(self, instr: BoundInstr[Self]):
        return self._writes

    @staticmethod
    def make(
        opcode: str,
        out_vars: Iterable[Var],
        *args: Value,
        continues: bool = True,
        jumps: bool = True,
        reads: Sequence[EffectBase] = (),
        writes: Sequence[EffectBase] = (),
        src_instr: type[InstrBase] | None = None,
    ) -> BoundInstr[RawInstr]:
        out_vars = tuple(out_vars)
        return RawInstr(
            opcode=opcode,
            in_types=TypeList(get_types(*args)),
            out_types=TypeList(get_types(*out_vars)),
            continues=continues,
            jumps=jumps,
            _reads=list(reads),
            _writes=list(writes),
            src_instr=src_instr,
        ).bind(out_vars, *args)

    @override
    def format(self, instr: BoundInstr[Self]) -> Text:
        ans = Text()
        ans.append(self.opcode, "ic10.raw_opcode")
        if len(instr.outputs_) > 0:
            ans += " ["
            ans += format_vals(instr.outputs_, self.out_types, sep=", ")
            ans += "]"
        ans += format_vals(instr.inputs_, self.in_types, sep=" ", prefix=" ")
        return ans

    def format_raw(self, instr: BoundInstr[Self], ctx: AsRawCtx) -> RawText:
        ans = Text()
        ans.append(self.opcode, "ic10.raw_opcode")
        for t, x in zip(self.out_types, instr.outputs_):
            ans += " "
            ans += Text("", "bold") + format_raw_val(x, ctx, t, instr.debug).text
        for t, x in zip(self.in_types, instr.inputs_):
            ans += " "
            ans += format_raw_val(x, ctx, t, instr.debug).text

        # if len(ans) < 40:
        #     ans += " " * (40 - len(ans))

        # ans += Text(f" # ({linenums.instr_lines[instr]})", "ic10.comment")
        # if loc_info := instr.debug.location_info_brief():
        #     ans += Text(" " + loc_info, "ic10.comment")

        ans += "\n"
        return RawText(ans)


class AsmInstrBase(InstrBase):
    """instr that lowers directly to an asm op"""

    opcode: str
    in_types: VarTS
    out_types: VarTS

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AsmInstrBase):
            return False
        return (
            type(self) == type(other)
            and self.in_types == other.in_types
            and self.out_types == other.out_types
        )

    @override
    def lower(self, instr: BoundInstr[Any]) -> Iterable[BoundInstr]:
        yield RawInstr(
            opcode=self.opcode,
            in_types=TypeList(self.in_types),
            out_types=TypeList(self.out_types),
            continues=instr.continues,
            jumps=True,
            _reads=list(instr.reads()),
            _writes=list(instr.writes()),
            src_instr=type(self),
        ).bind(instr.outputs, *instr.inputs)


class AsmInstrBinopSameType[T: VarT](AsmInstrBase):
    opcode: str

    def __init__(self, typ: type[T]) -> None:
        self.in_types = (typ, typ)
        self.out_types = (typ,)


################################################################################


class MoveBase[O: VarT = Any, I: VarT = Any](AsmInstrBase):
    jumps = False

    def __init__(self, in_typ: type[I], out_typ: type[O]) -> None:
        self.opcode = "move"
        self.in_types = (in_typ,)
        self.out_types = (out_typ,)

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        (ipt,) = instr.inputs_
        (opt,) = instr.outputs_

        if isinstance(ipt, Undef):
            if verbose.value >= 1:
                instr.debug.warn("undef assignment lowered as a no-op")
            return
        elif isinstance(ipt, Var) and ipt.reg.allocated == opt.reg.allocated:
            return
        else:
            yield from super().lower(instr)

    @override
    def regalloc_prefs(self, instr: BoundInstr[Self]) -> Iterable[RegallocPref]:
        (arg,) = instr.inputs_
        if isinstance(arg, Var):
            yield RegallocTie(arg, instr.outputs_[0], force=False)


class Move[T: VarT = Any](MoveBase[T, T]):
    def __init__(self, typ: type[T]) -> None:
        super().__init__(typ, typ)


class Transmute[O: VarT, I: VarT](MoveBase[O, I]):
    pass


class SplitLifetime[T: VarT = Any](MoveBase[T, T]):
    """
    isolate short-lived var,
    typically bc they have a reg preference,
    so it is not a good idea to extend its lifetime

    currently used in stack_chain_to_push_pop

    %ra fuses does not seem to need it atm
    (we currently use a mvar read/write to serve the same
    purpose; should work fine until lifetime analysis learns
    multi-level function calls)
    """

    def __init__(self, typ: type[T]) -> None:
        super().__init__(typ, typ)


################################################################################


@dataclass(frozen=True, eq=False)
class EffectPhantomLoc(EffectBase, ByIdMixin):
    id: int


class RemoveLabelProvenance[T: VarT = Any](MoveBase[T, T]):
    def __init__(self, typ: type[T], loc: EffectPhantomLoc) -> None:
        super().__init__(typ, typ)
        self.loc = loc

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return self.loc


class AddLabelProvenance[T: VarT = Any](MoveBase[T, T]):
    def __init__(self, typ: type[T], loc: EffectPhantomLoc) -> None:
        super().__init__(typ, typ)
        self.loc = loc

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return self.loc


################################################################################


class EndPlaceholder(InstrBase):
    in_types = ()
    out_types = ()
    continues = False


class Jump(AsmInstrBase):
    opcode = "j"
    in_types = (Label,)
    out_types = ()
    continues = False


jump = Jump().call


@dataclass
class UnreachableChecked(InstrBase):
    in_types = ()
    out_types = ()
    continues = False

    message: str | None = None

    @override
    def writes(self, instr: BoundInstr[Self]) -> None:
        return

    @override
    def lower(self, instr: BoundInstr[Self]) -> Never:
        instr.debug.error(
            self.message or "expected unreachable, but that can not be proved"
        ).throw()


def unreachable_checked(msg: str | None = None):
    with track_caller():
        UnreachableChecked(msg).call()


class BlackBox[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, typ: type[T]) -> None:
        self.opcode = "_BLACKBOX"
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

    @final
    def negate(self, instr: BoundInstr[PredicateBase]) -> BoundInstr[PredicateBase]:
        with clear_debug_info(), add_debug_info(instr.debug):
            return self.negate_impl(instr)

    def negate_impl(self, instr: BoundInstr[Any]) -> BoundInstr[PredicateBase]:
        raise NotImplementedError(f"negating {self}")

    def pred_for_cjump(self, instr: BoundInstr[Any]) -> BoundInstr[PredicateBase]:
        return instr


class PredVar(PredicateBase):
    """a predicate that is just a Bool"""

    opcode = "move"
    in_types = (bool,)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return Not().bind(instr.outputs_, *instr.inputs_)

    @override
    def pred_for_cjump(self, instr: BoundInstr[Self]) -> BoundInstr[PredicateBase]:
        return PredLE().bind(instr.outputs_, 1, instr.inputs_[0])


class Not(PredicateBase):
    opcode = "not"
    in_types = (bool,)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return PredVar().bind(instr.outputs_, *instr.inputs_)

    @override
    def pred_for_cjump(self, instr: BoundInstr[Self]) -> BoundInstr[PredicateBase]:
        return PredLT().bind(instr.outputs_, instr.inputs_[0], 1)


class Branch(InstrBase):
    in_types = (bool, Label, Label)
    out_types = ()
    continues = False


def branch(cond: InteralBool, on_true: InternalValLabel, on_false: InternalValLabel) -> None:
    return Branch().call(cond, on_true, on_false)


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

    consteval_fn = operator.add


class AddI(AsmInstrBase):
    opcode = "add"
    in_types = (int, int)
    out_types = (int,)

    consteval_fn = operator.add


class SubF(AsmInstrBase):
    opcode = "sub"
    in_types = (float, float)
    out_types = (float,)

    consteval_fn = operator.sub


class SubI(AsmInstrBase):
    opcode = "sub"
    in_types = (int, int)
    out_types = (int,)

    consteval_fn = operator.sub


class MulF(AsmInstrBase):
    opcode = "mul"
    in_types = (float, float)
    out_types = (float,)


class MulI(AsmInstrBase):
    opcode = "mul"
    in_types = (int, int)
    out_types = (int,)


class DivF(AsmInstrBase):
    opcode = "div"
    in_types = (float, float)
    out_types = (float,)


class Mod(AsmInstrBase):
    opcode = "mod"
    in_types = (int, int)
    out_types = (int,)


class OrB(AsmInstrBase):
    opcode = "or"
    in_types = (bool, bool)
    out_types = (bool,)


class OrI(AsmInstrBase):
    opcode = "or"
    in_types = (int, int)
    out_types = (int,)


class AndB(AsmInstrBase):
    opcode = "and"
    in_types = (bool, bool)
    out_types = (bool,)


class AndI(AsmInstrBase):
    opcode = "and"
    in_types = (int, int)
    out_types = (int,)


class XORB(AsmInstrBase):
    opcode = "xor"
    in_types = (bool, bool)
    out_types = (bool,)


class XORI(AsmInstrBase):
    opcode = "xor"
    in_types = (int, int)
    out_types = (int,)


class RShiftUnsigned(AsmInstrBase):
    opcode = "srl"
    in_types = (int, int)
    out_types = (int,)


class RShiftSigned(AsmInstrBase):
    opcode = "sra"
    in_types = (int, int)
    out_types = (int,)


class LShift(AsmInstrBase):
    opcode = "sll"
    in_types = (int, int)
    out_types = (int,)


class Min[T: VarT](AsmInstrBinopSameType[T]):
    opcode = "min"


class Max[T: VarT](AsmInstrBinopSameType[T]):
    opcode = "max"


################################################################################
# Math (non primitive)
################################################################################


class AbsI(AsmInstrBase):
    opcode = "abs"
    in_types = (int,)
    out_types = (int,)


class AbsF(AsmInstrBase):
    opcode = "abs"
    in_types = (float,)
    out_types = (float,)


################################################################################
# OTHER, PURE
################################################################################


class Select[T: VarT](AsmInstrBase):
    opcode = "select"

    def __init__(self, typ: type[T]) -> None:
        self.in_types = (bool, typ, typ)
        self.out_types = (typ,)


################################################################################
# PREDICATES
################################################################################


class PredLT(PredicateBase):
    opcode = "slt"
    in_types = (float, float)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        a, b = instr.inputs_
        return PredLE().bind(instr.outputs_, b, a)


class PredLE(PredicateBase):
    opcode = "sle"
    in_types = (float, float)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        a, b = instr.inputs_
        return PredLT().bind(instr.outputs_, b, a)


class PredEq[T: VarT = Any](PredicateBase):
    opcode = "seq"

    def __init__(self, typ: type[T]) -> None:
        self.typ = typ
        self.in_types = (typ, typ)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return PredNEq(self.typ).bind(instr.outputs_, *instr.inputs_)


class PredNEq[T: VarT = Any](PredicateBase):
    opcode = "sne"

    def __init__(self, typ: type[T]) -> None:
        self.typ = typ
        self.in_types = (typ, typ)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return PredEq(self.typ).bind(instr.outputs_, *instr.inputs_)


class PredNAN(PredicateBase):
    opcode = "snan"

    def __init__(self) -> None:
        self.in_types = (float,)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return PredNotNAN().bind(instr.outputs_, *instr.inputs_)


class PredNotNAN(PredicateBase):
    opcode = "snanz"

    def __init__(self) -> None:
        self.in_types = (float,)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return PredNAN().bind(instr.outputs_, *instr.inputs_)

    @override
    def pred_for_cjump(self, instr: BoundInstr[Self]) -> BoundInstr[PredicateBase]:
        err = instr.debug.error("there is no bnanz")
        err.note(
            f"this an internal error; fuse_blocks.py is not supposed to create {PredCondJump} with {self}"
        )
        err.throw()


################################################################################
# Bundles
################################################################################


@dataclass(eq=False)
class Bundle[*Ts = * tuple[BoundInstr[Any], ...]](InstrBase):
    # False, i -> var is inputs[i]
    # True, i -> var is outputs[i]
    var_info: list[tuple[bool, int]]

    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: TypeList[tuple[Var, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]

    children: list[tuple[InstrBase, tuple[int, ...], tuple[int, ...], DebugInfo]]

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

        vals: list[Value] = []
        out_vars: set[Var] = set()

        children: list[tuple[InstrBase, tuple[int, ...], tuple[int, ...], DebugInfo]] = []

        def handle_val(x: Value) -> int:
            if isinstance(x, Var) and x in vals:
                return vals.index(x)
            ans = len(vals)
            vals.append(x)
            return ans

        for instr in instrs:
            arg_idxs = tuple(handle_val(x) for x in instr.inputs)
            out_idxs = tuple(handle_val(x) for x in instr.outputs)
            out_vars |= set(instr.outputs)
            children.append((instr.instr, arg_idxs, out_idxs, instr.debug))

        var_info: list[tuple[bool, int]] = []
        inputs: list[Value] = []
        outputs: list[Var] = []

        for x in vals:
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

    @classmethod
    def from_block(cls, b: Block) -> BoundInstr[Self]:
        assert b.end.isinst(EndPlaceholder)
        with add_debug_info(b.debug):
            return cls.from_parts(*b.contents[:-1])  # pyright: ignore[reportArgumentType]

    @override
    def lower(self, instr: BoundInstr[Any], /) -> Iterable[BoundInstr]:
        return instr.unpack_untyped()

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

        comment = self.format_comment(instr)

        @group()
        def mk_group() -> Iterator[RenderableType]:
            if comment:
                yield comment
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

    @override
    def regalloc_prefs(self, instr: BoundInstr[Any], /) -> Iterable[RegallocPref]:
        for part in instr.unpack_untyped():
            yield from part.regalloc_prefs()


class PredBranch(
    Bundle[
        BoundInstr[PredicateBase],
        BoundInstr[Branch],
    ]
):
    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        pred, br = instr.unpack()
        predvar, t_l, f_l = br.inputs_

        if pred.isinst(PredNotNAN):
            pred = pred.instr.negate(pred)
            t_l, f_l = f_l, t_l

        yield PredCondJump.from_parts(
            pred,
            CondJump().bind((), predvar, t_l),
        )
        yield Jump().bind((), f_l)

    @override
    def regalloc_prefs(self, instr: BoundInstr[Self]) -> Iterable[RegallocPref]:
        pred, _br = instr.unpack()
        yield RegallocSkip(pred.outputs_[0])


class PredCondJump(
    Bundle[
        BoundInstr[PredicateBase],
        BoundInstr[CondJump],
    ]
):
    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        pred, cjump = instr.unpack()
        _predvar, label = cjump.inputs_

        pred = pred.instr.pred_for_cjump(pred)

        assert pred.instr.opcode.startswith("s")
        yield RawInstr.make("b" + pred.instr.opcode.removeprefix("s"), [], *pred.inputs, label)

    @override
    def regalloc_prefs(self, instr: BoundInstr[Self]) -> Iterable[RegallocPref]:
        pred, _br = instr.unpack()
        yield RegallocSkip(pred.outputs_[0])


class JumpAndLink(
    Bundle[
        BoundInstr[WriteMVar],
        BoundInstr[Jump],
        BoundInstr[EmitLabel],
    ]
):
    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        write_mv, jump, _emit_label = instr.unpack()
        if write_mv.instr.s.reg.allocated == Register.RA:
            yield RawInstr.make("jal", [], jump.inputs_[0])
        else:
            yield from super().lower(instr)


class CondJumpAndLink(
    Bundle[
        BoundInstr[WriteMVar],
        BoundInstr[PredicateBase],
        BoundInstr[CondJump],
        BoundInstr[EmitLabel],
    ]
):
    """
    FIXME:
    currently the semantic of CondJumpAndLink is that the
    "WriteMVar" happens unconditionally. this is not consistent
    with actual behavior; it:
    (1) extends the lifetime of the mvar
    (2) the mvar may be used later; making CondJumpAndLink
        causes the write to be possibly skipped and
        the wrong value be read later
        (i am not aware of any "reasonable" usage that will hit this problem)
    """

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        write_mv, pred, cjump, emit_label = instr.unpack()

        if write_mv.instr.s.reg.allocated == Register.RA:

            pred = pred.instr.pred_for_cjump(pred)

            opcode = pred.instr.opcode
            assert opcode.startswith("s")

            yield RawInstr.make(
                "b" + opcode.removeprefix("s") + "al", [], *pred.inputs, cjump.inputs_[1]
            )
        else:
            yield write_mv
            yield PredCondJump.from_parts(pred, cjump)
            yield emit_label

    @override
    def regalloc_prefs(self, instr: BoundInstr[Self]) -> Iterable[RegallocPref]:
        _write_mv, pred, _cjump, _emit_label = instr.unpack()
        yield RegallocSkip(pred.outputs_[0])


################################################################################
# STACK
################################################################################


class EffectStack(EffectBase):
    pass


class RawGet[T: VarT](AsmInstrBase):
    def __init__(self, typ: type[T]) -> None:
        self.opcode = "get"
        self.in_types = (PinType, int)
        self.out_types = (typ,)

    reads_ = EffectStack()


class RawPut[T: VarT](AsmInstrBase):
    def __init__(self, typ: type[T]) -> None:
        self.opcode = "put"
        self.in_types = (PinType, int, typ)
        self.out_types = ()

    writes_ = EffectStack()


def _format_stack_ptr(device: Value[PinType], addr: Value[int], offset: int) -> Text:
    ans = Text()

    if device != db_internal:
        ans += format_val(device, PinType)
        ans += ": "

    addr_part = format_val(addr, int)
    if offset != 0:
        addr_part = Text.assemble("(", addr_part, "+", str(offset), ")")

    ptr_text = Text("*", "ic10.pointer")
    ptr_text += addr_part
    ans += ptr_text

    return ans


class ReadStack[T: VarT = Any](InstrBase):
    jumps = False

    def __init__(self, typ: type[T], offset: int):
        self.in_types = (PinType, int)
        self.out_types = (typ,)
        self.offset = offset

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectStack()

    @override
    def format_expr_part(self, instr: BoundInstr[Self], /) -> Text:
        device, ptr = instr.inputs_
        return _format_stack_ptr(device, ptr, self.offset)

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        device, addr = instr.inputs_
        (out_var,) = instr.outputs_
        (typ,) = instr.out_types

        if isinstance(addr, Var):
            if self.offset == 0:
                addr_shifted = addr
            else:
                sp = Register.SP._mk_var(int)
                yield AddI().bind((sp,), addr, self.offset)
                addr_shifted = sp
        else:
            addr_shifted = ConstEval.addi(addr, self.offset)

        yield RawGet(typ).bind((out_var,), device, addr_shifted)


class WriteStack[T: VarT = Any](InstrBase):
    jumps = False

    def __init__(self, typ: type[T], offset: int):
        self.in_types = (PinType, int, typ)
        self.out_types = ()
        self.offset = offset

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectStack()

    @override
    def format(self, instr: BoundInstr[Self], /) -> Text:
        device, ptr, arg = instr.inputs_
        ptr_fmt = _format_stack_ptr(device, ptr, self.offset)

        ans = Text()
        ans += ptr_fmt
        ans += " = "
        ans += format_val(arg, self.in_types[2])
        return ans

    @override
    def lower(self, instr: BoundInstr[Self]):
        device, addr, arg = instr.inputs_
        _, _, typ = instr.in_types

        if isinstance(arg, Undef):
            return

        if isinstance(addr, Var):
            if self.offset == 0:
                addr_shifted = addr
            else:
                sp = Register.SP._mk_var(int)
                yield AddI().bind((sp,), addr, self.offset)
                addr_shifted = sp
        else:
            addr_shifted = ConstEval.addi(addr, self.offset)

        yield RawPut(typ).bind((), device, addr_shifted, arg)


class Push(
    Bundle[
        BoundInstr[WriteStack],
        BoundInstr[AddI],
    ]
):
    """
    write val to stack[db, addr],
    and then increment addr

    write must always be db and offset 0
    """

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        write, add = instr.unpack()

        device, addr, val = write.inputs_
        addr_, one_ = add.inputs_
        (addr_out,) = add.outputs_

        assert device == db_internal
        assert addr == addr_
        assert one_ == 1
        assert write.instr.offset == 0

        if (
            isinstance(addr, Var)
            and addr.reg.allocated == Register.SP
            and addr_out.reg.allocated == Register.SP
        ):
            return (RawInstr.make("push", (), val),)
        else:
            instr.debug.warn("failed to lower push to %sp")
            return super().lower(instr)


class Pop(
    Bundle[
        BoundInstr[SubI],
        BoundInstr[ReadStack],
    ]
):
    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        sub, read = instr.unpack()

        addr, one_ = sub.inputs_
        (addr_out,) = sub.outputs_
        device, addr_out_ = read.inputs_
        (var_out,) = read.outputs_

        assert device == db_internal
        assert addr_out_ == addr_out
        assert one_ == 1
        assert read.instr.offset == 0

        if (
            isinstance(addr, Var)
            and addr.reg.allocated == Register.SP
            and addr_out.reg.allocated == Register.SP
        ):
            return (RawInstr.make("pop", (var_out,)),)
        else:
            instr.debug.warn("failed to lower pop to %sp")
            return super().lower(instr)


class StackOpChain(
    Bundle[
        *tuple[
            BoundInstr[ReadStack] | BoundInstr[WriteStack],
            ...,
        ]
    ]
):
    """
    sequence of stack read and write on the same device at the same base addr

    currently is always a continuous read seg or write seg
    will prob change later
    """

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        parts = instr.unpack()

        if len(parts) <= 1:
            yield from parts
            return

        device, addr, *_val = parts[0].inputs_
        for p in parts[1:]:
            assert p.inputs_[0] == device
            assert p.inputs_[1] == addr

        if device != db_internal or not isinstance(addr, Var):
            yield from parts
            return

        if all(p.isinst(ReadStack) for p in parts):
            parts = [p.check_type(ReadStack) for p in parts]

            # we currently dont touch this after tracing
            # will prob change later
            for i, p in enumerate(parts):
                assert p.instr.offset == i

            sp = Register.SP._mk_var(int)
            yield AddI().bind((sp,), addr, len(parts))

            for p in reversed(parts):
                yield RawInstr.make("pop", (p.outputs_[0],))

            return

        if all(p.isinst(WriteStack) for p in parts):
            parts = [p.check_type(WriteStack) for p in parts]

            # we currently dont touch this after tracing
            # will prob change later
            for i, p in enumerate(parts):
                assert p.instr.offset == i

            sp = Register.SP._mk_var(int)
            yield Move(int).bind((sp,), addr)

            for p in parts:
                yield RawInstr.make("push", (), p.inputs_[2])

            return

        raise NotImplementedError()


################################################################################
# Raw Asm
################################################################################


@dataclass
class AsmBlock(InstrBase):
    lines: list[tuple[str, tuple[MVar | int, ...]]]

    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: tuple[()]

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        for opcode, vars in self.lines:
            yield RawInstr.make(
                opcode,
                (),
                *[instr.inputs_[x] if isinstance(x, int) else x._as_var_with_reg() for x in vars],
            )

    @override
    def reads(self, instr: BoundInstr[Self]):
        for _opcode, args in self.lines:
            for x in args:
                if not isinstance(x, int):
                    yield EffectMvar(x)

    @override
    def writes(self, instr: BoundInstr[Self]):
        for _opcode, args in self.lines:
            for x in args:
                if not isinstance(x, int):
                    yield EffectMvar(x)

    @override
    def format_with_anno(self, instr: BoundInstr[Self]) -> RenderableType:
        comment = self.format_comment(instr)

        @group()
        def mk_group() -> Iterator[RenderableType]:
            if len(comment) > 0:
                yield comment[2:]

            for opcode, vars in self.lines:
                ans = Text()
                ans += Text(opcode, "ic10.raw_opcode")
                for arg in vars:
                    ans += " "
                    if isinstance(arg, int):
                        ans += format_val(instr.inputs_[arg], self.in_types[arg])
                    else:
                        ans += arg._format()

                yield ans

        return Panel(
            mk_group(),
            title=Text(type(self).__name__, "ic10.jump"),
            # title=self.format(instr),
            title_align="left",
        )
