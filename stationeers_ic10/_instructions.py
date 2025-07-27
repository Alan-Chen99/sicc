from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import TypedDict
from typing import Unpack
from typing import overload
from typing import override

from rich.text import Text

from ._core import BoundInstr
from ._core import InstrBase
from ._core import InteralBool
from ._core import InternalFloat
from ._core import InternalValLabel
from ._core import Label
from ._core import TypeList
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VarTS
from ._core import format_val
from ._core import get_types


class EmitLabel(InstrBase):
    in_types = (Label,)
    out_types = ()

    jumps = False

    def format_with_args(self, l: Label) -> Text:
        return format_val(l) + ":"


@dataclass(frozen=True)
class RawInstr(InstrBase):
    opcode: str
    in_types: TypeList[tuple[Value, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]
    out_types: TypeList[tuple[Var, ...]]  # pyright: ignore[reportIncompatibleVariableOverride]

    continues: bool
    jumps: bool


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
        () = RawInstr(opcode, TypeList(), TypeList(get_types(*args)), **kwargs).emit(*args)
        return None
    else:
        (ans,) = RawInstr(
            opcode,
            TypeList((out_type,)),
            TypeList(get_types(*args)),
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
        return RawInstr(
            self.opcode,
            TypeList(self.out_types),
            TypeList(self.in_types),
            continues=self.continues,
            jumps=self.jumps,
        ).emit(*args)


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


class Jump(AsmInstrBase):
    opcode = "j"
    in_types = (Label,)
    out_types = ()
    continues = False


class UnreachableChecked(InstrBase):
    in_types = ()
    out_types = ()
    continues = False


class BlackBox[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, typ: type[T]) -> None:
        self.opcode = "move"
        self.in_types = (typ,)
        self.out_types = (typ,)


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
        raw_asm("br" + self.opcode.removeprefix("s"), None, *args, label)

    def lower_neg_cjump(self, *args: Value, label: InternalValLabel) -> None:
        (_out_var,), bound = self.negate()
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
    def __init__(self, pred: PredicateBase):
        self.base = pred

        self.in_types: TypeList[  # pyright: ignore[reportIncompatibleVariableOverride]
            tuple[InternalValLabel, InternalValLabel, *tuple[Value, ...]]
        ] = TypeList((Label, Label, *pred.in_types))
        self.out_types = ()
        self.continues = False

    def format_with_args(self, l_t: InternalValLabel, l_f: InternalValLabel, *args: Value) -> Text:
        ans = Text()
        ans.append(type(self).__name__, "ic10.jump")
        ans += " ["
        ans += self.base.format_with_args(*args)
        ans += "]"
        for x in [l_t, l_f]:
            ans += " "
            ans += format_val(x)
        return ans


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
