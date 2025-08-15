from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Self
from typing import TypeVar
from typing import overload
from typing import override

import rich.repr
from rich.pretty import pretty_repr

from ._api import Function
from ._api import Int
from ._api import Str
from ._api import UserValue
from ._api import Variable
from ._api import VarRead
from ._api import _get_type
from ._core import AnyType
from ._core import AsRaw
from ._core import AsRawCtx
from ._core import BoundInstr
from ._core import EffectRes
from ._core import PinType
from ._core import RawText
from ._core import Value
from ._core import VarT
from ._diagnostic import register_exclusion
from ._instructions import AsmInstrBase
from ._instructions import EffectExternal
from ._tree_utils import dataclass as optree_dataclass
from ._tree_utils import pytree
from ._utils import ReprAs
from ._utils import cast_unchecked
from ._utils import crc32

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
            LoadBatch,
            StoreBatch,
            LoadBatchNamed,
            StoreBatchNamed,
        ]:
            return RawText.str(self.name)
        return RawText.str(f"LogicType.{self.name}")


ValLogicTypeLike = UserValue[LogicType] | str
ValBatchMode = UserValue[BatchMode]


class Yield(AsmInstrBase):
    opcode = "yield"
    in_types = ()
    out_types = ()

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


yield_ = Function(Yield())


class Load[T: VarT](AsmInstrBase):
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "l"
        self.in_types = (PinType, LogicType)
        self.out_types = (out_type,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class Store[T: VarT](AsmInstrBase):
    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "s"
        self.in_types = (PinType, LogicType, in_type)
        self.out_types = ()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class LoadBatch[T: VarT](AsmInstrBase):
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "lb"
        self.in_types = (str, LogicType, BatchMode)
        self.out_types = (out_type,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class StoreBatch[T: VarT](AsmInstrBase):
    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "sb"
        self.in_types = (str, LogicType, in_type)
        self.out_types = ()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class LoadBatchNamed[T: VarT](AsmInstrBase):
    def __init__(self, out_type: type[T]) -> None:
        self.opcode = "lbn"
        self.in_types = (str, str, LogicType, BatchMode)
        self.out_types = (out_type,)

    @override
    def reads(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


class StoreBatchNamed[T: VarT](AsmInstrBase):
    def __init__(self, in_type: type[T]) -> None:
        self.opcode = "sbn"
        self.in_types = (str, str, LogicType, in_type)
        self.out_types = ()

    @override
    def writes(self, instr: BoundInstr[Self]) -> EffectRes:
        return EffectExternal()


@dataclass
class LiteralPin:
    """only for db, atm"""

    name: str


@optree_dataclass(eq=False, repr=False)
class Pin:
    _idx: Int | LiteralPin

    @staticmethod
    def db() -> Pin:
        return Pin(LiteralPin("db"))

    def __rich_repr__(self) -> rich.repr.Result:
        if isinstance(self._idx, LiteralPin):
            yield self._idx.name
        else:
            yield self._idx

    def __repr__(self) -> str:
        return pretty_repr(self)

    def _pin(self) -> UserValue[PinType]:
        if isinstance(self._idx, LiteralPin):
            return PinType(self._idx.name)
        return cast_unchecked(self._idx)

    def __getitem__(self, logic_type: ValLogicTypeLike) -> VarRead[Any]:
        return Function(Load(AnyType)).call(self._pin(), LogicType.create(logic_type))

    def __setitem__(self, logic_type: ValLogicTypeLike, val: UserValue) -> None:
        Function(Store(_get_type(val))).call(self._pin(), LogicType.create(logic_type), val)

    def __getattr__(self, name: str) -> VarRead[Any]:
        if name and name[0].isupper():
            return self[name]
        else:
            raise AttributeError()

    def __setattr__(self, name: str, val: UserValue) -> None:
        if name and name[0].isupper() and not hasattr(type(self), name):
            self[name] = val
        else:
            super().__setattr__(name, val)


_did_register_optree: set[type[DeviceBase[Any, Any]]] = set()

DT_co = TypeVar("DT_co", covariant=True, bound=Str, default=Str)
N_co = TypeVar("N_co", covariant=True, bound=Str | None, default=Str | None)


@dataclass(eq=False, repr=False)
class DeviceBase(Generic[DT_co, N_co]):
    device_type: DT_co
    name: N_co
    default_batchmode: BatchMode

    def __init__(
        self,
        *,
        _device_type: DT_co,
        _name: N_co,
        _default_batchmode: BatchMode = BatchMode.AVG,
    ) -> None:
        if not type(self) in _did_register_optree:
            pytree.register_node_class()(type(self))
            _did_register_optree.add(type(self))

        self.device_type = _device_type
        self.name = _name
        self.default_batchmode = _default_batchmode

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.device_type
        if self.name is not None:
            yield self.name

    def __repr__(self) -> str:
        return pretty_repr(self)

    @overload
    def __getitem__(self, logic_type: ValLogicTypeLike) -> DeviceLogicType[Any, Self]: ...
    @overload
    def __getitem__(self, logic_type: tuple[ValLogicTypeLike, ValBatchMode]) -> Variable[Any]: ...

    def __getitem__(self, logic_type: ValLogicTypeLike | tuple[ValLogicTypeLike, ValBatchMode]):
        if isinstance(logic_type, tuple):
            logic_type, bm = logic_type
            return DeviceLogicType(
                self, LogicType.create(logic_type), AnyType, self.default_batchmode
            ).get(bm)
        return DeviceLogicType(self, LogicType.create(logic_type), AnyType, self.default_batchmode)

    def __setitem__(self, logic_type: ValLogicTypeLike, val: UserValue):
        self[logic_type].set(val)

    def _getattr(self, name: str) -> DeviceLogicType[Any, Self]:
        if name and name[0].isupper():
            return self[name]
        else:
            raise AttributeError()

    def _setattr(self, name: str, val: UserValue) -> None:
        if name and name[0].isupper() and not hasattr(type(self), name):
            self[name].set(val)
        else:
            super().__setattr__(name, val)

    if not TYPE_CHECKING:
        __getattr__ = _getattr
        __setattr__ = _setattr

    def tree_flatten(self):
        return (self.device_type, self.name, self.default_batchmode), None, None

    @classmethod
    def tree_unflatten(cls, metadata: Any, children: Any) -> Self:
        device_type, name, default_batchmode = children
        ans = cls.__new__(cls)
        DeviceBase.__init__(
            ans, _device_type=device_type, _name=name, _default_batchmode=default_batchmode
        )
        return ans


class Device[D: Str = Str, N: Str | None = Str | None](DeviceBase[D, N]):
    def __init__(self, device_type: D, name: N = None) -> None:
        super().__init__(_device_type=device_type, _name=name)


T = TypeVar("T", bound=VarT, default=Any)
D_co = TypeVar("D_co", covariant=True, bound=DeviceBase, default=DeviceBase)


@dataclass(eq=False)
class DeviceLogicTypeRead(Generic[T, D_co], VarRead[T]):
    device: D_co
    logic_type: UserValue[LogicType]
    typ: type[T]
    default_batchmode: BatchMode

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.device
        yield self.logic_type

    def __repr__(self) -> str:
        return pretty_repr(self)

    def get(self, mode: ValBatchMode | None = None) -> VarRead[T]:
        if mode is None:
            mode = self.default_batchmode

        if self.device.name is None:
            return Function(LoadBatch(self.typ)).call(
                self.device.device_type, self.logic_type, mode
            )
        return Function(LoadBatchNamed(self.typ)).call(
            self.device.device_type, self.device.name, self.logic_type, mode
        )

    @property
    def avg(self) -> VarRead[T]:
        return self.get(BatchMode.AVG)

    @property
    def sum(self) -> VarRead[T]:
        return self.get(BatchMode.SUM)

    @property
    def min(self) -> VarRead[T]:
        return self.get(BatchMode.MIN)

    @property
    def max(self) -> VarRead[T]:
        return self.get(BatchMode.MAX)

    @override
    def _read(self) -> Value[T]:
        return self.get()._read()

    @override
    def _get_type(self) -> type[T]:
        return self.typ


class DeviceLogicType(Generic[T, D_co], DeviceLogicTypeRead[T, D_co]):
    def set(self, val: UserValue[T]) -> None:
        typ = _get_type(val) if self.typ == AnyType else self.typ
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
        return DeviceLogicTypeRead(obj, self.logic_type, self.typ, obj.default_batchmode)

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
        return DeviceLogicType(obj, self.logic_type, self.typ, obj.default_batchmode)

    def __set__(self, obj: DeviceBase, val: UserValue[T]) -> None:
        assert self.logic_type != None
        DeviceLogicType(obj, self.logic_type, self.typ, obj.default_batchmode).set(val)


def mk_field[T: VarT](typ: type[T] = AnyType, logic_type: LogicType | None = None) -> FieldDescW[T]:
    return FieldDescW(typ, logic_type)


def mk_field_ro[T: VarT](
    typ: type[T] = AnyType, logic_type: LogicType | None = None
) -> FieldDesc[T]:
    return FieldDesc(typ, logic_type)


class DeviceTyped(DeviceBase[str]):
    _hash: int

    def __init__(self, name: Str | None = None, *, default_batchmode: BatchMode = BatchMode.AVG):
        device_type = f"Structure{type(self).__name__}"
        assert crc32(device_type) == self._hash
        super().__init__(_device_type=device_type, _name=name, _default_batchmode=default_batchmode)

    def __rich_repr__(self) -> rich.repr.Result:
        if self.name is not None:
            yield self.name

    PrefabHash: FieldDesc[int] = mk_field_ro(int)
    ReferenceId: FieldDesc[int] = mk_field_ro(int)
    NameHash: FieldDesc[int] = mk_field_ro(int)
