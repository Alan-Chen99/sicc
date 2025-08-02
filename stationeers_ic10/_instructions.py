from __future__ import annotations

from dataclasses import dataclass
from typing import Any  # autoflake: skip
from typing import Callable
from typing import ClassVar
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
from rich.text import Text

from ._core import Block
from ._core import BoundInstr
from ._core import EffectBase
from ._core import InstrBase
from ._core import InteralBool
from ._core import InternalFloat
from ._core import InternalValLabel
from ._core import Label
from ._core import MVar
from ._core import TypeList
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VarTS
from ._core import WriteMVar
from ._core import format_val
from ._core import get_type
from ._core import get_types
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import clear_debug_info
from ._utils import mk_ordered_set
from ._utils import safe_cast

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

    negate: ClassVar[Callable[..., tuple[tuple[Var[bool]], BoundInstr]]]

    def lower_neg(self, *args: Value) -> Var[bool]:
        (out_var,), bound = self.negate(*args)
        bound.emit()
        return out_var

    def lower_cjump(self, *args: Value, label: InternalValLabel) -> None:
        assert self.opcode.startswith("s")
        raw_asm("b" + self.opcode.removeprefix("s"), None, *args, label)

    def lower_neg_cjump(self, *args: Value, label: InternalValLabel) -> None:
        (_out_var,), bound = self.negate(*args)
        assert isinstance(bound.instr, PredicateBase)
        bound.instr.lower_cjump(*bound.inputs, label=label)


class PredVar(PredicateBase):
    """a predicate that is just a Bool"""

    opcode = "move"
    in_types = (bool,)

    @override
    def negate(self, a: InteralBool):
        return Not().create_bind(a)

    @override
    def lower_cjump(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, a: InteralBool, label: InternalValLabel
    ) -> None:
        from ._instructions import PredLE

        PredLE().lower_cjump(1, a, label=label)


class Not(PredicateBase):
    opcode = "not"
    in_types = (bool,)

    @override
    def negate(self, a: InteralBool):
        return PredVar().create_bind(a)

    @override
    def lower_cjump(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, a: InteralBool, label: InternalValLabel
    ) -> None:
        from ._instructions import PredLT

        PredLT().lower_cjump(a, 1, label=label)


class Branch(InstrBase):
    in_types = (bool, Label, Label)
    out_types = ()
    continues = False


class PredBranch(InstrBase):
    def __init__(self, pred: PredicateBase):
        assert pred.out_types == (bool,)
        self.base = pred

        self.in_types: TypeList[  # pyright: ignore[reportIncompatibleVariableOverride]
            tuple[InternalValLabel, InternalValLabel, *tuple[Value, ...]]
        ] = TypeList((Label, Label, *pred.in_types))
        self.out_types = (bool,)

    @override
    def bundles(self, instr: BoundInstr[Self]) -> Iterator[BoundInstr]:
        l_t, l_f, *args = instr.inputs_
        (pred_var,) = instr.outputs_
        yield self.base.bind((pred_var,), *args)  # pyright: ignore
        yield Branch().bind((), pred_var, l_t, l_f)

    # @override
    # def format_with_args(self, l_t: InternalValLabel, l_f: InternalValLabel, *args: Value) -> Text:
    #     ans = Text()
    #     ans.append(type(self).__name__, "ic10.jump")
    #     ans += " ["
    #     ans += self.base.format_with_args(*args)
    #     ans += "]"
    #     for x in [l_t, l_f]:
    #         ans += " "
    #         ans += format_val(x)
    #     return ans

    # def lower(self, l_t: InternalValLabel, l_f: InternalValLabel, *args: Value) -> None:
    #     CJump(self.base, jump_on=True).call(l_t, *args)
    #     Jump().call(l_f)

    #     # assert self.opcode.startswith("s")
    #     # raw_asm("b" + self.opcode.removeprefix("s"), None, *args, label)


class JumpAndLink(InstrBase):
    """
    contains link_reg, which is a mvar with reg preference to ra

    takes (return_label, call_label)
    equivalent to

    link_reg := return_label
    jump call_label

    jump to label and set link_reg to
    """

    def __init__(self, link_reg: MVar):
        self.link_reg = link_reg
        self.in_types = (Label, Label)
        self.out_types = ()

    @override
    def bundles(self, instr: BoundInstr[Self]):
        return_label, call_label = instr.inputs_
        yield WriteMVar(self.link_reg).bind((), return_label)
        yield Jump().bind((), call_label)


class CondJumpAndLink(InstrBase):
    def __init__(self, pred: PredicateBase, link_reg: MVar):
        self.link_reg = link_reg

        assert pred.out_types == (bool,)
        self.pred = pred

        self.in_types = safe_cast(
            TypeList[tuple[InternalValLabel, InternalValLabel, *tuple[Value, ...]]],
            TypeList((Label, Label, *pred.in_types)),
        )
        self.out_types = ()

    @override
    def bundles(self, instr: BoundInstr[Self]):
        func_label, ret_label, *pred_args = instr.inputs_
        return (
            WriteMVar(self.link_reg).bind((), ret_label),
            CondJump(self.pred).bind((), func_label, *pred_args),
            EmitLabel().bind((), ret_label),
        )

    # @override
    # def format_with_args(self, target: InternalValLabel, *args: Value) -> Text:
    #     ans = Text()
    #     ans.append(type(self).__name__, "ic10.jump")
    #     ans += " "
    #     if self.jump_on == False:
    #         ans.append("NOT", "ic10.jump")
    #     ans += "["
    #     ans += self.base.format_with_args(*args)
    #     ans += "] "
    #     ans += format_val(target)
    #     return ans


class CondJump(InstrBase):
    """
    jumps to a label if pred evaluates equal to "jump_on"; otherwise continue

    tracing does not emit this; only used after optimize when concating blocks
    """

    def __init__(self, pred: PredicateBase):
        assert pred.out_types == (bool,)
        self.base = pred
        # self.jump_on = jump_on

        self.in_types: TypeList[  # pyright: ignore[reportIncompatibleVariableOverride]
            tuple[InternalValLabel, *tuple[Value, ...]]
        ] = TypeList((Label, *pred.in_types))
        self.out_types = ()
        self.continues = True

    # @override
    # def format_with_args(self, target: InternalValLabel, *args: Value) -> Text:
    #     ans = Text()
    #     ans.append(type(self).__name__, "ic10.jump")
    #     ans += " "
    #     if self.jump_on == False:
    #         ans.append("NOT", "ic10.jump")
    #     ans += "["
    #     ans += self.base.format_with_args(*args)
    #     ans += "] "
    #     ans += format_val(target)
    #     return ans

    # @override
    # def lower(self, label: InternalValLabel, *args: Value) -> None:
    #     assert False
    #     if self.jump_on == True:
    #         return self.base.lower_cjump(*args, label=label)
    #     else:
    #         return self.base.lower_neg_cjump(*args, label=label)


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
    def negate(self, a: InternalFloat, b: InternalFloat):
        return PredLE().create_bind(b, a)


class PredLE(PredicateBase):
    opcode = "sle"
    in_types = (float, float)

    @override
    def negate(self, a: InternalFloat, b: InternalFloat):
        return PredLT().create_bind(b, a)


@dataclass
class Isolate(InstrBase):
    """for debugging and testing only"""

    # atm inputs must be plain values (cannot be Var). this may change in future

    # False, i -> var is inputs[i]
    # True, i -> var is outputs[i]
    var_info: list[tuple[bool, int]]

    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: TypeList[tuple[Var, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]

    children: list[tuple[InstrBase, tuple[int, ...], tuple[int, ...], DebugInfo]]

    @override
    def bundles(self, instr: BoundInstr[Self]) -> Iterable[BoundInstr]:
        vars = [instr.outputs[x] if is_out else instr.inputs[x] for is_out, x in self.var_info]
        for child, args, outs, dbg in self.children:
            arg_vals = tuple(vars[i] for i in args)
            out_vars = tuple(cast(Var, vars[i]) for i in outs)
            with clear_debug_info(), add_debug_info(dbg):
                yield child.bind_untyped(out_vars, *arg_vals)

    @staticmethod
    def from_list(instrs: list[BoundInstr]) -> BoundInstr[Isolate]:
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

        ans = Isolate(
            var_info=var_info,
            in_types=TypeList((get_type(x) for x in inputs)),
            out_types=TypeList((x.type for x in outputs)),
            children=children,
        )

        return ans.bind_untyped(tuple(outputs), *inputs)

    @staticmethod
    def from_block(b: Block) -> BoundInstr[Isolate]:
        assert b.end.isinst(EndPlaceholder)
        with add_debug_info(b.debug):
            return Isolate.from_list(b.contents[:-1])
