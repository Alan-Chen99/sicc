from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Self
from typing import Sequence
from typing import cast
from typing import final
from typing import override

from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.console import RenderableType
from rich.console import group
from rich.panel import Panel
from rich.text import Text

from ._core import AsRawCtx
from ._core import Block
from ._core import BoundInstr
from ._core import EffectBase
from ._core import EffectRes
from ._core import InstrBase
from ._core import InternalValLabel
from ._core import Label
from ._core import RawText
from ._core import RegallocExtend
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
from ._core import WriteMVar
from ._core import format_instr_list
from ._core import format_raw_val
from ._core import format_val
from ._core import get_type
from ._core import get_types
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import clear_debug_info
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
            ans += format_val(instr.outputs_[0])
            for x in instr.outputs_[1:]:
                ans += ", "
                ans += format_val(x)
            ans += "]"

        for x in instr.inputs_:
            ans += " "
            ans += format_val(x)

        return ans

    def format_raw(self, instr: BoundInstr[Self], ctx: AsRawCtx) -> RawText:
        ans = Text()
        ans.append(self.opcode, "ic10.raw_opcode")
        for x in instr.outputs_:
            ans += " "
            ans += Text("", "bold") + format_raw_val(x, ctx).text
        for x in instr.inputs_:
            ans += " "
            ans += format_raw_val(x, ctx).text

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

    @override
    def lower(self, instr: BoundInstr[Any]) -> Iterable[BoundInstr]:
        yield RawInstr.make(
            self.opcode,
            instr.outputs,
            *instr.inputs,
            continues=instr.continues,
            src_instr=type(self),
        )


################################################################################


class MoveBase[O: VarT, I: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, in_typ: type[I], out_typ: type[O]) -> None:
        self.opcode = "move"
        self.in_types = (in_typ,)
        self.out_types = (out_typ,)

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        (ipt,) = instr.inputs_
        (opt,) = instr.outputs_

        if ipt == Undef.undef():
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


class SubF(AsmInstrBase):
    opcode = "sub"
    in_types = (float, float)
    out_types = (float,)


class SubI(AsmInstrBase):
    opcode = "sub"
    in_types = (int, int)
    out_types = (int,)


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


class PredEq[T: VarT](PredicateBase):
    opcode = "seq"

    def __init__(self, typ: type[T]) -> None:
        self.typ = typ
        self.in_types = (typ, typ)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return PredNEq(self.typ).bind(instr.outputs_, *instr.inputs_)


class PredNEq[T: VarT](PredicateBase):
    opcode = "sne"

    def __init__(self, typ: type[T]) -> None:
        self.typ = typ
        self.in_types = (typ, typ)

    @override
    def negate_impl(self, instr: BoundInstr[Self]):
        return PredEq(self.typ).bind(instr.outputs_, *instr.inputs_)


################################################################################
# Bundles
################################################################################


@dataclass
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


@dataclass
class AsmBlockInner(InstrBase):
    """
    a block:
    add [a] a 1
    breq a 0 +2
    add [b] a x
    add [a] a 2

    will have:
    inputs: [a_init, b_init, 1, x, 2] (init vals of outs followed by consts)
    outputs: [a_out, b_out]

    x is considerred a "const" bc it is never written
    a, b are written so they are out vars

    note that even though it appears that b is never read, it actually "is":
    if the branch skips the b assignment, the prev value of b will be the result
    so it is neccessary for us to have "b_init" as a arg

    a_init and a_out must be allocated the same register, etc
    to make sure that this is always possible, we wrap this with a bundle "AsmBlock"
    which insert a Move on each argument
    """

    # opcode, out_vars, in_vars
    lines: list[tuple[str, tuple[int, ...]]]

    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: TypeList[tuple[Var, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def n_vars(self) -> int:
        return len(self.out_types)

    @override
    def regalloc_prefs(self, instr: BoundInstr[Self]) -> Iterable[RegallocPref]:
        for in_const in instr.inputs_[self.n_vars :]:
            if isinstance(in_const, Var):
                yield RegallocExtend(in_const)

        for out_var, out_init in zip(instr.outputs_, instr.inputs_[: self.n_vars]):
            if isinstance(out_init, Var):
                yield RegallocTie(out_var, out_init, force=True)

    @override
    def lower(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        for out_var, out_init in zip(instr.outputs_, instr.inputs_[: self.n_vars]):
            assert isinstance(out_init, Var)
            assert out_init.reg.allocated == out_var.reg.allocated

        for opcode, vars in self.lines:
            yield RawInstr.make(
                opcode,
                (),
                *[instr.outputs_[x] if x < self.n_vars else instr.inputs_[x] for x in vars],
            )

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectBase:
        return EffectExternal()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectBase:
        return EffectExternal()

    @override
    def format_with_anno(self, instr: BoundInstr[Self]) -> RenderableType:
        comment = self.format_comment(instr)

        @group()
        def mk_group() -> Iterator[RenderableType]:
            if len(comment) > 0:
                yield comment[2:]

            def fmt(v: int) -> Text:
                if v < self.n_vars:
                    return Text(f"%{chr(ord('a') + v)}", "bold")
                return format_val(instr.inputs_[v])

            for i in range(self.n_vars):
                ans = Text()
                ans += fmt(i)
                ans += Text.assemble("(", format_val(instr.outputs_[i]), ")")
                ans += " := "
                ans += format_val(instr.inputs_[i])

                yield ans

            for opcode, vars in self.lines:
                ans = Text()
                ans += Text(opcode, "ic10.raw_opcode")
                for arg in vars:
                    ans += " "
                    ans += fmt(arg)

                yield ans

        return Panel(
            mk_group(),
            title=Text(type(self).__name__, "ic10.jump"),
            # title=self.format(instr),
            title_align="left",
        )


class AsmBlock(Bundle[*tuple[BoundInstr[AsmBlockInner | Move], ...]]):
    pass
