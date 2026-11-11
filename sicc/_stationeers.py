from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import Generic
from typing import Literal
from typing import Self
from typing import TypeVar
from typing import overload
from typing import override

import rich.repr
from rich.pretty import pretty_repr

from ._api import Bool
from ._api import Function
from ._api import Int
from ._api import Str
from ._api import UserValue
from ._api import VarRead
from ._api import _get_type
from ._api import and_
from ._api import not_equal
from ._api import or_
from ._api import rshift_signed
from ._api import sub
from ._api import transmute
from ._api import xor
from ._core import AnyType
from ._core import AsRaw
from ._core import AsRawCtx
from ._core import BoundInstr
from ._core import EffectRes
from ._core import PinType
from ._core import RawText
from ._core import Value
from ._core import VarT
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import clear_debug_info
from ._diagnostic import debug_info
from ._diagnostic import register_exclusion
from ._instructions import AsmInstrBase
from ._instructions import EffectExternal
from ._instructions import LShift
from ._tree_utils import field as optree_field  # pyright: ignore[reportUnknownVariableType]
from ._tree_utils import optree_dataclass
from ._tree_utils import pytree
from ._utils import ReprAs
from ._utils import cast_unchecked
from ._utils import crc32
from ._utils import transpile_setattr_default
from ._utils import transpile_setitem_default

register_exclusion(__file__)


class BatchMode(Enum):
    AVG = 0
    SUM = 1
    MIN = 2
    MAX = 3

    def as_raw(self, ctx: AsRawCtx) -> RawText:
        return RawText.str(repr(self.value))

    def __repr__(self) -> str:
        return self.name


class Color(Enum):
    Blue = 0
    Gray = 1
    Green = 2
    Orange = 3
    Red = 4
    Yellow = 5
    White = 6
    Black = 7
    Brown = 8
    Khaki = 9
    Pink = 10
    Purple = 11

    def __repr__(self) -> str:
        return self.name

    def as_raw(self, ctx: AsRawCtx) -> RawText:
        return RawText.str(repr(self.value))


@optree_dataclass(kw_only=True)
class LiteralPin:
    """
    internal class; only for db, atm

    This exist because PinType() is a single scalar; LiteralPin is a pytree with no leaves.
    One cannot pass "db" to a function that wants pin number; LiteralPin() makes these structurally different

    FIXME: consider making "db" and "d<n>" independent types in the compiler
    """

    _name: Literal["db"] = optree_field(pytree_node=False)

    @staticmethod
    def _db() -> LiteralPin:
        return LiteralPin(_name="db")

    def __repr__(self) -> str:
        return "db"


@dataclass(frozen=True)
class LogicType(AsRaw):
    name: str

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def create(val: ValLogicTypeLike) -> UserValue[LogicType]:
        if isinstance(val, str):
            return LogicType(val)
        return val

    def as_raw(self, ctx: AsRawCtx) -> RawText:
        if ctx.instr is not None and ctx.instr.instr.src_instr in [
            Load,
            Store,
            LoadById,
            StoreById,
            LoadBatch,
            StoreBatch,
            LoadBatchNamed,
            StoreBatchNamed,
        ]:
            return RawText.str(self.name)
        return RawText.str(f"LogicType.{self.name}")


@dataclass(frozen=True)
class SlotType(AsRaw):
    name: str

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def create(val: ValSlotTypeLike) -> UserValue[SlotType]:
        if isinstance(val, str):
            return SlotType(val)
        return val

    def as_raw(self, ctx: AsRawCtx) -> RawText:
        if ctx.instr is not None and ctx.instr.instr.src_instr in [
            LoadSlot,
            LoadBatchSlot,
            LoadBatchNamedSlot,
            StoreSlot,
            StoreBatchSlot,
        ]:
            return RawText.str(self.name)
        # FIXME: this works or no?
        return RawText.str(f"LogicSlotType.{self.name}")


ValLogicTypeLike = UserValue[LogicType] | str
ValSlotTypeLike = UserValue[SlotType] | str
ValBatchMode = UserValue[BatchMode]


class Yield(AsmInstrBase):
    opcode = "yield"
    in_types = ()
    out_types = ()

    reads_ = EffectExternal()
    writes_ = EffectExternal()


yield_ = Function(Yield())


class Sleep(AsmInstrBase):
    opcode = "sleep"
    in_types = (float,)
    out_types = ()

    reads_ = EffectExternal()
    writes_ = EffectExternal()


sleep = Function(Sleep())


class Load[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "l"
        self.in_types = (PinType, LogicType)
        self.out_types = (out_type,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class Store[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "s"
        self.in_types = (PinType, LogicType, in_type)
        self.out_types = ()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class LoadBatch[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "lb"
        self.in_types = (str, LogicType, BatchMode)
        self.out_types = (out_type,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class StoreBatch[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "sb"
        self.in_types = (str, LogicType, in_type)
        self.out_types = ()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class LoadBatchNamed[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "lbn"
        self.in_types = (str, str, LogicType, BatchMode)
        self.out_types = (out_type,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class StoreBatchNamed[T: VarT](AsmInstrBase):
    jumps = False

    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "sbn"
        self.in_types = (str, str, LogicType, in_type)
        self.out_types = ()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class LoadById[T: VarT](AsmInstrBase):
    jumps = False

    # ld r? referenceId logicType
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "ld"
        self.in_types = (int, LogicType)
        self.out_types = (out_type,)

    reads_ = EffectExternal()


class StoreById[T: VarT](AsmInstrBase):
    jumps = False

    # sd referenceId logicType r?
    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "sd"
        self.in_types = (int, LogicType, in_type)
        self.out_types = ()

    writes_ = EffectExternal()


def _valid_logic_attr(name: str, parent: object) -> bool:
    if name and name[0].isupper() and not hasattr(type(parent), name):
        return True
    return False


_did_register_optree: set[type[DeviceBase[Any, Any]]] = set()

DT_co = TypeVar("DT_co", covariant=True, bound=Str | None, default=Str | None)
N_co = TypeVar("N_co", covariant=True, bound=Str | None, default=Str | None)


@dataclass(repr=False)
class DeviceBase(Generic[DT_co, N_co]):
    device_type: DT_co
    name: N_co
    default_batchmode: BatchMode
    _pin: Int | LiteralPin | None
    # the device's ReferenceId; set for a device addressed by it (ld/sd/getd/...)
    _ref_id: Int | None

    def __init__(
        self,
        *,
        _device_type: DT_co,
        _name: N_co,
        _default_batchmode: BatchMode = BatchMode.AVG,
        _pin: Int | LiteralPin | None = None,
        _ref_id: Int | None = None,
    ) -> None:
        if not type(self) in _did_register_optree:
            pytree.register_node_class()(type(self))
            _did_register_optree.add(type(self))
        assert _pin is None or _ref_id is None

        self.device_type = _device_type
        self.name = _name
        self.default_batchmode = _default_batchmode
        self._pin = _pin
        self._ref_id = _ref_id

    def __rich_repr__(self) -> rich.repr.Result:
        if self._pin is not None:
            yield self._pin
        elif self._ref_id is not None:
            yield "ref_id", self._ref_id
        else:
            yield self.device_type
            if self.name is not None:
                yield self.name

    def __repr__(self) -> str:
        return pretty_repr(self)

    def _pin_value(self) -> UserValue[PinType]:
        """Convert pin to UserValue[PinType] for instructions."""
        assert self._pin is not None
        if isinstance(self._pin, LiteralPin):
            return PinType(_name=self._pin._name)
        return cast_unchecked(self._pin)

    def _id_value(self) -> Int:
        """
        The ReferenceId, for the instructions that address a device only by pin or ReferenceId
        (ss, getd, putd, clrd). A batch-addressed device reads its ReferenceId with the default
        batch mode, which is the ReferenceId only if exactly one device matches.
        """
        assert self._pin is None
        if self._ref_id is not None:
            return self._ref_id
        return DeviceLogicTypeRead(self, LogicType("ReferenceId"), int)

    @overload
    def __getitem__(self, logic_type: ValLogicTypeLike) -> DeviceLogicType[Any, Self]: ...
    @overload
    def __getitem__(self, logic_type: tuple[ValLogicTypeLike, ValBatchMode]) -> VarRead[Any]: ...

    def __getitem__(self, logic_type: ValLogicTypeLike | tuple[ValLogicTypeLike, ValBatchMode]):
        if isinstance(logic_type, tuple):
            logic_type, bm = logic_type
            return DeviceLogicType(self, LogicType.create(logic_type), AnyType).get(mode=bm)
        return DeviceLogicType(self, LogicType.create(logic_type), AnyType)

    def __setitem__(self, logic_type: ValLogicTypeLike, val: UserValue):
        self[logic_type].set(val)

    def _getattr(self, name: str) -> DeviceLogicType[Any, Self]:
        if _valid_logic_attr(name, self):
            return self[name]
        else:
            raise AttributeError(name)

    def _setattr(self, name: str, val: UserValue) -> None:
        if _valid_logic_attr(name, self):
            self[name].set(val)
        else:
            super().__setattr__(name, val)

    if not TYPE_CHECKING:
        __getattr__ = _getattr
        __setattr__ = _setattr

    __transpile_setattr__ = transpile_setattr_default
    __transpile_setitem__ = transpile_setitem_default

    def tree_flatten(self) -> Any:
        children = (self.device_type, self.name, self._pin, self._ref_id)
        return children, (self.default_batchmode,), None

    @classmethod
    def tree_unflatten(cls, metadata: Any, children: Any) -> Self:
        device_type, name, pin, ref_id = children
        (default_batchmode,) = metadata
        ans = cls.__new__(cls)
        DeviceBase.__init__(
            ans,
            _device_type=device_type,
            _name=name,
            _default_batchmode=default_batchmode,
            _pin=pin,
            _ref_id=ref_id,
        )
        return ans

    @property
    def StaticPrefabHash(self) -> DT_co:
        return self.device_type

    @property
    def slots(self) -> Slots[Slot]:
        """The device's slots; `slots[i]["Occupied"]` reads LogicSlotType Occupied of slot i."""
        return Slots(self, Slot)

    def as_generic_device(self) -> Device:
        assert self.device_type is not None
        return Device(
            name=self.name,
            device_type=self.device_type,
        )

    def as_generic_device_keep_type(self) -> Self:
        return cast_unchecked(self.as_generic_device())


class Pin(DeviceBase[None, None]):
    """
    Access a device via pin number.

    Uses direct l/s instructions instead of batch operations.

    Examples:
        d0 = Pin(0)
        d0.Temperature  # uses 'l' instruction
        d0.Setting = 100  # uses 's' instruction
    """

    def __init__(self, idx: PinSpec) -> None:
        pin = _parse_pin(idx)
        super().__init__(_device_type=None, _name=None, _pin=pin)

    @staticmethod
    def db() -> Pin:
        return Pin("db")

    @property
    def memory(self) -> MemoryReadWrite[Any]:
        """
        The device's memory, untyped: `memory[address]` (get), `memory[address] = value` (put),
        `memory.clear()` (clr); e.g. the stack of the chip in an IC housing.
        """
        return MemoryReadWrite(self, AnyType, None)

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> VarRead[Any]: ...
        def __setattr__(self, name: str, val: UserValue) -> None: ...


PinSpec = Pin | Int | Literal["db"]


def _parse_pin(pin: PinSpec | None) -> Int | LiteralPin | None:
    """Normalize a user-facing pin spec to the internal representation.

    Returns:
        - None: for "no pin" (batch-mode device access)
        - LiteralPin("db"): for databank pin
        - Int/UserValue[int]: for numeric pin indices

    Notes:
        We intentionally do *not* return PinType here. PinType is the scalar runtime
        value used by instructions; LiteralPin exists to keep "db" as a distinct
        pytree leafless node.
    """

    if pin is None:
        return None
    if isinstance(pin, Pin):
        # Pin stores pin spec in its _pin field already.
        return pin._pin
    if isinstance(pin, str):
        assert pin == "db"
        return LiteralPin._db()
    # int / Variable[int] / other Int-like UserValue
    return pin


################################################################################
# PARAMS / OUTPUTS
################################################################################


T = TypeVar("T", bound=VarT, default=Any)
D_co = TypeVar("D_co", covariant=True, bound=DeviceBase, default=DeviceBase)


@dataclass(eq=False)
class DeviceLogicTypeRead(Generic[T, D_co], VarRead[T]):
    device: D_co
    logic_type: UserValue[LogicType]
    typ: type[T]
    _debug: DebugInfo = field(default_factory=debug_info)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.device
        yield self.logic_type

    def __repr__(self) -> str:
        return pretty_repr(self)

    @overload
    def get(self, *, mode: ValBatchMode | None = None) -> VarRead[T]: ...
    @overload
    def get[T1: VarT](self, *, mode: ValBatchMode | None = None, typ: type[T1]) -> VarRead[T1]: ...

    def get(
        self, *, mode: ValBatchMode | None = None, typ: type[VarT] | None = None
    ) -> VarRead[VarT]:
        # Pin-based access: use direct Load instruction
        if self.device._pin is not None:
            return Function(Load(typ or self.typ)).call(self.device._pin_value(), self.logic_type)
        if self.device._ref_id is not None:
            return Function(LoadById(typ or self.typ)).call(self.device._ref_id, self.logic_type)

        assert self.device.device_type is not None
        if mode is None:
            mode = self.device.default_batchmode

        if self.device.name is None:
            return Function(LoadBatch(typ or self.typ)).call(
                self.device.device_type, self.logic_type, mode
            )
        return Function(LoadBatchNamed(typ or self.typ)).call(
            self.device.device_type, self.device.name, self.logic_type, mode
        )

    @property
    def avg(self) -> VarRead[T]:
        return self.get(mode=BatchMode.AVG)

    @property
    def sum(self) -> VarRead[T]:
        return self.get(mode=BatchMode.SUM)

    @property
    def min(self) -> VarRead[T]:
        return self.get(mode=BatchMode.MIN)

    @property
    def max(self) -> VarRead[T]:
        return self.get(mode=BatchMode.MAX)

    @override
    def _read(self) -> Value[T]:
        with clear_debug_info(), add_debug_info(self._debug):
            return self.get()._read()

    @override
    def _get_type(self) -> type[T]:
        return self.typ


class DeviceLogicType(Generic[T, D_co], DeviceLogicTypeRead[T, D_co]):
    def set(self, val: UserValue[T]) -> None:
        typ = _get_type(val) if self.typ == AnyType else self.typ

        # Pin-based access: use direct Store instruction
        if self.device._pin is not None:
            return Function(Store(typ)).call(self.device._pin_value(), self.logic_type, val)
        if self.device._ref_id is not None:
            return Function(StoreById(typ)).call(self.device._ref_id, self.logic_type, val)
        assert self.device.device_type is not None

        if self.device.name is None:
            return Function(StoreBatch(typ)).call(self.device.device_type, self.logic_type, val)

        return Function(StoreBatchNamed(typ)).call(
            self.device.device_type, self.device.name, self.logic_type, val
        )


@dataclass
class FieldDesc[T: VarT = Any]:
    typ: type[T]
    logic_type: LogicType | None = None

    def __set_name__(self, owner: Any, name: str):
        if self.logic_type is None:
            self.logic_type = LogicType(name)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...
    @overload
    def __get__[D: DeviceBase](self, obj: D, objtype: type[D]) -> DeviceLogicTypeRead[T, D]: ...
    def __get__[D: DeviceBase](self, obj: D | None, objtype: type[D]):
        if obj is None:
            return self
        assert self.logic_type != None
        return DeviceLogicTypeRead(obj, self.logic_type, self.typ)

    def _set_fail(self, obj: DeviceBase, val: UserValue[T]) -> None:
        raise AttributeError("write to read-only field")

    if not TYPE_CHECKING:
        __set__ = _set_fail

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.logic_type
        yield ReprAs(self.typ.__name__)


class FieldDescW[T: VarT = Any](FieldDesc[T]):
    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...
    @overload
    def __get__[D: DeviceBase](self, obj: D, objtype: type[D]) -> DeviceLogicType[T, D]: ...
    def __get__[D: DeviceBase](self, obj: D | None, objtype: type[D]):
        if obj is None:
            return self
        assert self.logic_type != None
        return DeviceLogicType(obj, self.logic_type, self.typ)

    def __set__(self, obj: DeviceBase, val: UserValue[T]) -> None:
        assert self.logic_type != None
        DeviceLogicType(obj, self.logic_type, self.typ).set(val)


def mk_field[T: VarT](typ: type[T] = AnyType, logic_type: LogicType | None = None) -> FieldDescW[T]:
    return FieldDescW(typ, logic_type)


def mk_field_ro[T: VarT](
    typ: type[T] = AnyType, logic_type: LogicType | None = None
) -> FieldDesc[T]:
    return FieldDesc(typ, logic_type)


################################################################################
# SLOTS
################################################################################


class Slots[S: Slot = Slot]:
    """`device.slots`; `slots[i]` is slot i of the device."""

    # Final, not a dataclass field: Slots[S] is then covariant in S, so a device's typed slots
    # override DeviceBase.slots
    device: Final[DeviceBase]
    slot_cls: Final[type[S]]

    def __init__(self, device: DeviceBase, slot_cls: type[S]) -> None:
        self.device = device
        self.slot_cls = slot_cls

    def tree_flatten(self) -> Any:
        return (self.device,), self.slot_cls, None

    @classmethod
    def tree_unflatten(cls, metadata: Any, children: Any) -> Self:
        (device,) = children
        return cls(device, metadata)

    def __getitem__(self, idx: Int) -> S:
        return self.slot_cls(self.device, idx)


pytree.register_node_class()(Slots)


class Slot:
    """
    Slot `idx` of `device`. `slot["Occupied"]` is LogicSlotType Occupied, untyped.

    Typed devices give their slots a subclass with an attribute, typed, for each LogicSlotType
    that their slots have; with a compile-time index, using one that the slot lacks is an error.
    """

    device: DeviceBase
    idx: Int

    def __init__(self, device: DeviceBase, idx: Int) -> None:
        self.device = device
        self.idx = idx

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        pytree.register_node_class()(cls)

    def tree_flatten(self) -> Any:
        return (self.device, self.idx), None, None

    @classmethod
    def tree_unflatten(cls, metadata: Any, children: Any) -> Self:
        device, idx = children
        return cls(device, idx)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.device
        yield self.idx

    def __repr__(self) -> str:
        return pretty_repr(self)

    @overload
    def __getitem__(self, logic_type: ValSlotTypeLike) -> SlotField: ...
    @overload
    def __getitem__(self, logic_type: tuple[ValSlotTypeLike, ValBatchMode]) -> VarRead[Any]: ...

    def __getitem__(self, logic_type: ValSlotTypeLike | tuple[ValSlotTypeLike, ValBatchMode]):
        if isinstance(logic_type, tuple):
            logic_type, bm = logic_type
            return SlotField(self.device, self.idx, SlotType.create(logic_type), AnyType).get(
                mode=bm
            )
        return SlotField(self.device, self.idx, SlotType.create(logic_type), AnyType)

    def __setitem__(self, logic_type: ValSlotTypeLike, val: UserValue) -> None:
        self[logic_type].set(val)

    __transpile_setattr__ = transpile_setattr_default
    __transpile_setitem__ = transpile_setitem_default


pytree.register_node_class()(Slot)


class LoadSlot[T: VarT](AsmInstrBase):
    jumps = False

    # ls r? device slotIndex logicSlotType
    # device: a pin, or (dev_type int) a ReferenceId
    def __init__(self, out_type: type[T], dev_type: type[VarT] = PinType) -> None:
        self.opcode = "ls"
        self.in_types = (dev_type, int, SlotType)
        self.out_types = (out_type,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class LoadBatchSlot[T: VarT](AsmInstrBase):
    jumps = False

    # lbs r? deviceHash slotIndex logicSlotType batchMode
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "lbs"
        self.in_types = (str, int, SlotType, BatchMode)
        self.out_types = (out_type,)

    reads_ = EffectExternal()


class LoadBatchNamedSlot[T: VarT](AsmInstrBase):
    jumps = False

    # lbns r? deviceHash nameHash slotIndex logicSlotType batchMode
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "lbns"
        self.in_types = (str, str, int, SlotType, BatchMode)
        self.out_types = (out_type,)

    reads_ = EffectExternal()


class StoreSlot[T: VarT](AsmInstrBase):
    jumps = False

    # ss device slotIndex logicSlotType r?
    # device: a pin, or (dev_type int) a ReferenceId
    def __init__(self, in_type: type[T], dev_type: type[VarT] = PinType) -> None:
        self.opcode = "ss"
        self.in_types = (dev_type, int, SlotType, in_type)
        self.out_types = ()

    writes_ = EffectExternal()


class StoreBatchSlot[T: VarT](AsmInstrBase):
    jumps = False

    # sbs deviceHash slotIndex logicSlotType r?
    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "sbs"
        self.in_types = (str, int, SlotType, in_type)
        self.out_types = ()

    writes_ = EffectExternal()


@dataclass(eq=False)
class SlotFieldRead[T: VarT = Any](VarRead[T]):
    device: DeviceBase
    idx: Int
    logic_type: UserValue[SlotType]
    typ: type[T]

    _debug: DebugInfo = field(default_factory=debug_info)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.device
        yield self.idx
        yield self.logic_type

    def __repr__(self) -> str:
        return pretty_repr(self)

    @overload
    def get(self, *, mode: ValBatchMode | None = None) -> VarRead[T]: ...
    @overload
    def get[T1: VarT](self, *, mode: ValBatchMode | None = None, typ: type[T1]) -> VarRead[T1]: ...

    def get(
        self, *, mode: ValBatchMode | None = None, typ: type[VarT] | None = None
    ) -> VarRead[VarT]:
        # Pin-based access: use direct LoadSlot instruction
        if self.device._pin is not None:
            return Function(LoadSlot(typ or self.typ)).call(
                self.device._pin_value(), self.idx, self.logic_type
            )
        if self.device._ref_id is not None:
            return Function(LoadSlot(typ or self.typ, int)).call(
                self.device._ref_id, self.idx, self.logic_type
            )

        assert self.device.device_type is not None
        if mode is None:
            mode = self.device.default_batchmode

        if self.device.name is None:
            return Function(LoadBatchSlot(typ or self.typ)).call(
                self.device.device_type, self.idx, self.logic_type, mode
            )
        return Function(LoadBatchNamedSlot(typ or self.typ)).call(
            self.device.device_type, self.device.name, self.idx, self.logic_type, mode
        )

    @property
    def avg(self) -> VarRead[T]:
        return self.get(mode=BatchMode.AVG)

    @property
    def sum(self) -> VarRead[T]:
        return self.get(mode=BatchMode.SUM)

    @property
    def min(self) -> VarRead[T]:
        return self.get(mode=BatchMode.MIN)

    @property
    def max(self) -> VarRead[T]:
        return self.get(mode=BatchMode.MAX)

    @override
    def _read(self) -> Value[T]:
        with clear_debug_info(), add_debug_info(self._debug):
            return self.get()._read()

    @override
    def _get_type(self) -> type[T]:
        return self.typ


class SlotField[T: VarT = Any](SlotFieldRead[T]):
    def set(self, val: UserValue[T]) -> None:
        """
        ss on a pin; sbs (every device of the type) on a batch device without a name; ss by
        ReferenceId otherwise, as IC10 has no named batch slot write.
        """
        typ = _get_type(val) if self.typ == AnyType else self.typ

        if self.device._pin is not None:
            return Function(StoreSlot(typ)).call(
                self.device._pin_value(), self.idx, self.logic_type, val
            )
        if self.device._ref_id is None and self.device.name is None:
            assert self.device.device_type is not None
            return Function(StoreBatchSlot(typ)).call(
                self.device.device_type, self.idx, self.logic_type, val
            )
        return Function(StoreSlot(typ, int)).call(
            self.device._id_value(), self.idx, self.logic_type, val
        )


def _check_slot_index(slot: Slot, logic_type: SlotType, write: bool) -> None:
    """With a compile-time index, check that slot `idx` of the device has the LogicSlotType."""
    if not isinstance(slot.idx, int):
        return
    device = slot.device
    assert isinstance(device, DeviceTyped)
    if not 0 <= slot.idx < device._slot_count:
        raise TypeError(
            f"{device._device_type_from_cls_name()} has slots 0 to {device._slot_count - 1};"
            f" there is no slot {slot.idx}"
        )
    name = logic_type.name
    if write:
        ok = slot.idx in device._slot_write[name]
    else:
        ok = name not in device._slot_only or slot.idx in device._slot_only[name]
    if not ok:
        raise TypeError(
            f"{device._device_type_from_cls_name()}: slot {slot.idx} has no"
            f" {'writable ' if write else ''}{name}"
        )


@dataclass
class SlotFieldDesc[T: VarT = Any]:
    typ: type[T]
    logic_type: SlotType | None = None

    def __set_name__(self, owner: Any, name: str):
        if self.logic_type is None:
            self.logic_type = SlotType(name)

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...
    @overload
    def __get__(self, obj: Slot, objtype: type) -> SlotFieldRead[T]: ...
    def __get__(self, obj: Slot | None, objtype: type):
        if obj is None:
            return self
        assert self.logic_type is not None
        _check_slot_index(obj, self.logic_type, write=False)
        return SlotFieldRead(obj.device, obj.idx, self.logic_type, self.typ)

    def _set_fail(self, obj: Slot, val: UserValue[T]) -> None:
        raise AttributeError("write to read-only slot field")

    if not TYPE_CHECKING:
        __set__ = _set_fail

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.logic_type
        yield ReprAs(self.typ.__name__)


class SlotFieldDescW[T: VarT = Any](SlotFieldDesc[T]):
    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...
    @overload
    def __get__(self, obj: Slot, objtype: type) -> SlotField[T]: ...
    def __get__(self, obj: Slot | None, objtype: type):
        if obj is None:
            return self
        assert self.logic_type is not None
        _check_slot_index(obj, self.logic_type, write=False)
        return SlotField(obj.device, obj.idx, self.logic_type, self.typ)

    def __set__(self, obj: Slot, val: UserValue[T]) -> None:
        assert self.logic_type is not None
        _check_slot_index(obj, self.logic_type, write=True)
        SlotField(obj.device, obj.idx, self.logic_type, self.typ).set(val)


def mk_slot_field[T: VarT](typ: type[T]) -> SlotFieldDescW[T]:
    return SlotFieldDescW(typ)


def mk_slot_field_ro[T: VarT](typ: type[T]) -> SlotFieldDesc[T]:
    return SlotFieldDesc(typ)


################################################################################


class Device[D: Str = Str, N: Str | None = Str | None](DeviceBase[D, N]):
    def __init__(
        self,
        device_type: D,
        name: N = None,
        default_batchmode: BatchMode = BatchMode.AVG,
    ) -> None:
        super().__init__(_device_type=device_type, _name=name, _default_batchmode=default_batchmode)

    PrefabHash: FieldDesc[str] = mk_field_ro(str)
    NameHash: FieldDesc[str] = mk_field_ro(str)
    ReferenceId: FieldDesc[int] = mk_field_ro(int)

    def exist(self) -> VarRead[bool]:
        # >=1 devices with this PrefabHash
        return self.PrefabHash.get(mode=BatchMode.SUM, typ=int) != 0

    def is_unique(self) -> VarRead[bool]:
        # ==1 devices with this PrefabHash
        return self.PrefabHash.sum == self.StaticPrefabHash


class DeviceTyped(Device[str, Str | None]):
    _hash: int
    # for prefab names other than "Structure" + class name, e.g. "StopWatch"
    _prefab_name: str | None = None
    # number of slots
    _slot_count: int = 0
    # LogicSlotType -> the slots that have it, for those that not every slot has
    _slot_only: dict[str, tuple[int, ...]] = {}
    # LogicSlotType -> the slots where it is writable
    _slot_write: dict[str, tuple[int, ...]] = {}

    @classmethod
    def _device_type_from_cls_name(cls) -> str:
        if cls._prefab_name is not None:
            return cls._prefab_name
        return f"Structure{cls.__name__}"

    def __init__(
        self,
        name: Str | None = None,
        *,
        default_batchmode: BatchMode = BatchMode.AVG,
        pin: PinSpec | None = None,
        ref_id: Int | None = None,
    ):
        """
        The devices of this type, or those named `name`, through batch instructions (lb, lbn,
        ...); or the device on `pin` (l, s, ...); or the device whose ReferenceId is `ref_id`
        (ld, sd, ...).
        """
        device_type = self._device_type_from_cls_name()
        assert crc32(device_type) == self._hash

        pin_ = _parse_pin(pin)
        if (pin_ is not None) + (ref_id is not None) + (name is not None) > 1:
            raise TypeError("give at most one of name, pin and ref_id")

        DeviceBase.__init__(
            self,
            _device_type=device_type,
            _name=name,
            _default_batchmode=default_batchmode,
            _pin=pin_,
            _ref_id=ref_id,
        )

    def __rich_repr__(self) -> rich.repr.Result:
        if self._pin is not None:
            yield "pin", self._pin
        if self._ref_id is not None:
            yield "ref_id", self._ref_id
        if self.name is not None:
            yield self.name

    def tree_flatten(self) -> Any:
        return (self.name, self._pin, self._ref_id), (self.default_batchmode,), None

    @classmethod
    def tree_unflatten(cls, metadata: Any, children: Any) -> Self:
        name, pin, ref_id = children
        (default_batchmode,) = metadata
        ans = cls.__new__(cls)
        DeviceBase.__init__(
            ans,
            _device_type=cls._device_type_from_cls_name(),
            _name=name,
            _default_batchmode=default_batchmode,
            _pin=pin,
            _ref_id=ref_id,
        )
        return ans


################################################################################
# MEMORY
################################################################################


class DeviceGet[T: VarT](AsmInstrBase):
    jumps = False

    # get r? device address
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "get"
        self.in_types = (PinType, int)
        self.out_types = (out_type,)

    reads_ = EffectExternal()


class DeviceGetById[T: VarT](AsmInstrBase):
    jumps = False

    # getd r? referenceId address
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "getd"
        self.in_types = (int, int)
        self.out_types = (out_type,)

    reads_ = EffectExternal()


class DevicePut[T: VarT](AsmInstrBase):
    jumps = False

    # put device address value
    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "put"
        self.in_types = (PinType, int, in_type)
        self.out_types = ()

    writes_ = EffectExternal()


class DevicePutById[T: VarT](AsmInstrBase):
    jumps = False

    # putd referenceId address value
    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "putd"
        self.in_types = (int, int, in_type)
        self.out_types = ()

    writes_ = EffectExternal()


class DeviceClear(AsmInstrBase):
    jumps = False
    opcode = "clr"
    in_types = (PinType,)
    out_types = ()

    writes_ = EffectExternal()


class DeviceClearById(AsmInstrBase):
    jumps = False
    opcode = "clrd"
    in_types = (int,)
    out_types = ()

    writes_ = EffectExternal()


@optree_dataclass
class _MemoryBase[T: VarT = Any]:
    device: DeviceBase
    typ: type[T] = optree_field(pytree_node=False)
    # number of addresses; None when it depends on the device (a circuit housing's chip)
    size: int | None = optree_field(pytree_node=False)

    def _pin(self) -> UserValue[PinType] | None:
        """The pin for get/put/clr; None to use getd/putd/clrd."""
        if self.device._pin is None:
            return None
        if isinstance(self.device._pin, LiteralPin):
            raise TypeError(
                "the memory of db is this program's own stack; use stack_var or Pointer for it"
            )
        return self.device._pin_value()

    def _check_address(self, address: Int) -> None:
        if isinstance(address, int) and self.size is not None and not 0 <= address < self.size:
            raise TypeError(f"address {address} is outside the memory, 0 to {self.size - 1}")


@optree_dataclass
class MemoryRead[T: VarT = Any](_MemoryBase[T]):
    """`device.memory` of a device whose memory IC10 can read: `memory[address]` (get, getd)."""

    def __getitem__(self, address: Int) -> VarRead[T]:
        self._check_address(address)
        if (pin := self._pin()) is not None:
            return Function(DeviceGet(self.typ)).call(pin, address)
        return Function(DeviceGetById(self.typ)).call(self.device._id_value(), address)


@optree_dataclass
class MemoryWrite[T: VarT = Any](_MemoryBase[T]):
    """
    `device.memory` of a device whose memory IC10 can write: `memory[address] = value` (put,
    putd) and `memory.clear()` (clr, clrd).
    """

    def __setitem__(self, address: Int, value: UserValue[T]) -> None:
        self._check_address(address)
        typ = _get_type(value) if self.typ == AnyType else self.typ
        if (pin := self._pin()) is not None:
            return Function(DevicePut(typ)).call(pin, address, value)
        Function(DevicePutById(typ)).call(self.device._id_value(), address, value)

    def clear(self) -> None:
        """Set every address to 0."""
        if (pin := self._pin()) is not None:
            return Function(DeviceClear()).call(pin)
        Function(DeviceClearById()).call(self.device._id_value())

    __transpile_setitem__ = transpile_setitem_default


@optree_dataclass
class MemoryReadWrite[T: VarT = Any](MemoryRead[T], MemoryWrite[T]):
    """`device.memory`: `memory[address]`, `memory[address] = value` and `memory.clear()`."""


################################################################################
# DEVICE INSTRUCTIONS
################################################################################


def instr_pack(opcode: int, *fields: tuple[Bool | Int | Str, int, int, bool]) -> Int:
    """
    A memory word holding a device instruction: `opcode` in bits 0-7, and each field
    (value, first bit, bit count, signed) above. A str is a hash. A compile-time value must fit
    its field. A runtime value is truncated to its field, except in the topmost field, which
    takes it whole: a negative value there sign-extends over the unused bits, as the game packs
    its own words.
    """
    top = max((pos for _, pos, _, _ in fields), default=None)
    const = opcode
    runtime: list[VarRead[int]] = []
    for val, pos, bits, signed in fields:
        mask = (1 << bits) - 1
        whole = pos == top
        if isinstance(val, (bool, int, str)):
            v = crc32(val) if isinstance(val, str) else int(val)
            lo, hi = (-(1 << (bits - 1)), 1 << (bits - 1)) if signed else (0, 1 << bits)
            if not lo <= v < hi:
                kind = "a signed" if signed else "an unsigned"
                raise TypeError(f"{val!r} does not fit in {kind} {bits}-bit field")
            const |= (v if whole else v & mask) << pos
        else:
            # bool casts to int implicitly; a hash does not
            v: Int = transmute(val, int) if _get_type(val) == str else cast_unchecked(val)
            runtime.append(Function(LShift()).call(v if whole else and_(v, mask), pos))
    word: Int = const
    for part in runtime:
        word = or_(word, part)
    return word


def instr_unpack[T: VarT](
    word: Int, pos: int, bits: int, signed: bool, typ: type[T], *, whole: bool = False
) -> VarRead[T]:
    """
    Field (first bit `pos`, `bits` bits) of a memory word holding a device instruction. `whole`:
    the bits above the field copy its sign (signed) or are 0 (unsigned), so shifting suffices.
    """
    v = rshift_signed(word, pos)
    if not whole:
        v = and_(v, (1 << bits) - 1)
        if signed:
            sign = 1 << (bits - 1)
            v = sub(xor(v, sign), sign)
    if typ == bool:
        return cast_unchecked(not_equal(v, 0))
    return transmute(v, typ)


def instr_opcode(word: Int) -> VarRead[int]:
    """OP_CODE (bits 0-7) of a memory word holding a device instruction."""
    return and_(word, 0xFF)


################################################################################


@dataclass(frozen=True)
class GasPort:
    # for debug/printing mostly
    device: DeviceBase
    suffix: str

    Combustion: DeviceLogicTypeRead[bool]
    Pressure: DeviceLogicTypeRead[float]
    Temperature: DeviceLogicTypeRead[float]
    TotalMoles: DeviceLogicTypeRead[float]

    RatioCarbonDioxide: DeviceLogicTypeRead[float]
    RatioHelium: DeviceLogicTypeRead[float]
    RatioHydrazine: DeviceLogicTypeRead[float]
    RatioHydrochloricAcid: DeviceLogicTypeRead[float]
    RatioHydrogen: DeviceLogicTypeRead[float]
    RatioLiquidAlcohol: DeviceLogicTypeRead[float]
    RatioLiquidCarbonDioxide: DeviceLogicTypeRead[float]
    RatioLiquidHydrazine: DeviceLogicTypeRead[float]
    RatioLiquidHydrochloricAcid: DeviceLogicTypeRead[float]
    RatioLiquidHydrogen: DeviceLogicTypeRead[float]
    RatioLiquidMethane: DeviceLogicTypeRead[float]
    RatioLiquidNitrogen: DeviceLogicTypeRead[float]
    RatioLiquidNitrousOxide: DeviceLogicTypeRead[float]
    RatioLiquidOxygen: DeviceLogicTypeRead[float]
    RatioLiquidOzone: DeviceLogicTypeRead[float]
    RatioLiquidPollutant: DeviceLogicTypeRead[float]
    RatioLiquidSilanol: DeviceLogicTypeRead[float]
    RatioLiquidSodiumChloride: DeviceLogicTypeRead[float]
    RatioMethane: DeviceLogicTypeRead[float]
    RatioNitrogen: DeviceLogicTypeRead[float]
    RatioNitrousOxide: DeviceLogicTypeRead[float]
    RatioOxygen: DeviceLogicTypeRead[float]
    RatioOzone: DeviceLogicTypeRead[float]
    RatioPollutant: DeviceLogicTypeRead[float]
    RatioPollutedWater: DeviceLogicTypeRead[float]
    RatioSilanol: DeviceLogicTypeRead[float]
    RatioSteam: DeviceLogicTypeRead[float]
    RatioWater: DeviceLogicTypeRead[float]

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.device
        if self.suffix:
            yield self.suffix

    def __repr__(self) -> str:
        return pretty_repr(self)
