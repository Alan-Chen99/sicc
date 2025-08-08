from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import ClassVar
from typing import Self
from typing import cast
from typing import overload
from typing import override

from rich.pretty import pretty_repr

from ._api import Function
from ._api import Str
from ._api import UserValue
from ._api import VarRead
from ._core import AnyType
from ._core import AsLiteral
from ._core import BoundInstr
from ._core import EffectRes
from ._core import VarT
from ._diagnostic import register_exclusion
from ._instructions import AsmInstrBase
from ._instructions import EffectExternal
from ._tree_utils import dataclasses
from ._tree_utils import pytree

register_exclusion(__file__)


class BatchMode(Enum):
    AVG = "Average"
    SUM = "Sum"
    MIN = "Minimum"
    MAX = "Maximum"

    def as_literal(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.name


@dataclass
class LogicType(AsLiteral):
    name: str

    def __repr__(self) -> str:
        return f"LogicType.{self.name}"

    @staticmethod
    def create(val: ValLogicTypeLike) -> UserValue[LogicType]:
        if isinstance(val, str):
            return LogicType(val)
        return val

    def as_literal(self) -> str:
        return f"LogicType.{self.name}"


class Pin(AsLiteral):
    # TODO
    pass


ValLogicTypeLike = UserValue[LogicType] | str
ValBatchMode = UserValue[BatchMode]


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


_did_register_optree: set[type[DeviceBase]] = set()


@dataclass
class DeviceBase:
    device_type: Str
    name: Str | None = None

    _did_register_optree: ClassVar[bool] = False

    def __init__(
        self,
        name: Str | None = None,
        *,
        device_type: Str | None = None,
        pin: Pin | None = None,
    ) -> None:
        if not type(self) in _did_register_optree:
            pytree.register_node_class()(type(self))
            _did_register_optree.add(type(self))

        if pin is not None:
            raise NotImplementedError()

        if device_type is None:
            assert type(self) != DeviceBase
            device_type = type(self).__name__

        self.device_type = device_type
        self.name = name

    def __rich_repr__(self):
        yield self.device_type
        if self.name is not None:
            yield self.name

    def __repr__(self) -> str:
        return pretty_repr(self)

    @overload
    def __getitem__(self, logic_type: ValLogicTypeLike) -> DeviceLogicType[Any, Self]: ...
    @overload
    def __getitem__(self, logic_type: tuple[ValLogicTypeLike, ValBatchMode]) -> VarRead[Any]: ...

    def __getitem__(self, logic_type: ValLogicTypeLike | tuple[ValLogicTypeLike, ValBatchMode]):
        if isinstance(logic_type, tuple):
            logic_type, bm = logic_type
            return DeviceLogicType(self, LogicType.create(logic_type), AnyType).get(bm)
        return DeviceLogicType(self, LogicType.create(logic_type), AnyType)

    def __getattr__(self, name: str) -> DeviceLogicType[Any, Self]:
        return self[name]

    def tree_flatten(self):
        return (self.device_type, self.name), None, None

    @classmethod
    def tree_unflatten(cls, metadata: Any, children: Any) -> Self:
        device_type, name = cast(tuple[Str, Str | None], children)
        ans = cls.__new__(cls)
        DeviceBase.__init__(ans, device_type=device_type, name=name)
        return ans


@dataclasses.dataclass
class DeviceLogicType[T: VarT = Any, D: DeviceBase = DeviceBase]:
    device: D
    logic_type: UserValue[LogicType]
    typ: type[T] = dataclasses.field(pytree_node=False)  # pyright: ignore[reportUnknownMemberType]

    def get(self, mode: ValBatchMode) -> VarRead[T]:
        if self.device.name is None:
            return Function(LoadBatch(self.typ)).call(
                self.device.device_type, self.logic_type, mode
            )
        return Function(LoadBatchNamed(self.typ)).call(
            self.device.device_type, self.device.name, self.logic_type, mode
        )

    def set(self, val: UserValue[T]) -> None:
        if self.device.name is None:
            return Function(StoreBatch(self.typ)).call(
                self.device.device_type, self.logic_type, val
            )
        return Function(StoreBatchNamed(self.typ)).call(
            self.device.device_type, self.device.name, self.logic_type, val
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


@dataclass
class FieldDesc[T: VarT]:
    typ: type[T]
    logic_type: LogicType | None = None

    def __set_name__(self, owner: Any, name: str):
        if self.logic_type is None:
            self.logic_type = LogicType(name)

    def __get__[D: DeviceBase](self, obj: D, objtype: type[D]) -> DeviceLogicType[T, D]:
        assert self.logic_type != None
        assert obj is not None
        return DeviceLogicType(obj, self.logic_type, self.typ)

    def __set__(self, obj: DeviceBase, val: UserValue[T]) -> None:
        assert self.logic_type != None
        DeviceLogicType(obj, self.logic_type, self.typ).set(val)


class Device(DeviceBase):
    def __init__(self, device_type: Str, name: Str | None = None) -> None:
        super().__init__(name, device_type=device_type)


def mk_field[T: VarT](typ: type[T], logic_type: LogicType | None = None) -> FieldDesc[T]:
    return FieldDesc(typ, logic_type)
