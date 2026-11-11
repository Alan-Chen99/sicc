"""
Generate sicc/devices.py from game_data/stationeers_logic.json.

The JSON is dumped from the game by game_data/dump.sh. Besides the device classes, with their
logic fields, slots and memory, devices.py gets a class for each device instruction set
(SorterInstruction, ...), and docstrings from the game's Stationpedia text. This script also
checks that the hand-written sicc._stationeers.Color and sicc._stationeers.GasPort match the
dumped game enums.

Usage: python scripts/build_types.py
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from typing import TypedDict

from sicc._stationeers import Color
from sicc._stationeers import GasPort
from sicc._utils import crc32

DUMP_FILE = Path(__file__).parent / "game_data" / "stationeers_logic.json"
OUT_FILE = Path(__file__).parent.parent / "sicc" / "devices.py"

PORT_SUFFIXES = ["", "Input", "Input2", "Output", "Output2"]

# exposed on sicc._stationeers.Device for every device
DEVICE_FIELDS = {"NameHash", "PrefabHash", "ReferenceId"}

# The values GetLogicValue returns for the field on every device that has it, read off the
# decompiled game (Stationeers 0.2.6429; ilspycmd on the Assembly-CSharp.dll of the server that
# game_data/dump.sh installs). The game's own field descriptions are not reliable: Activate is
# "1 if device is activated, otherwise 0", yet StructureHarvie stores 2 in it.
# Port variants of the gas fields (PressureInput, ...) are typed in generate().
FIELD_TYPES: dict[str, str] = {
    "Color": "_Color",
    **dict.fromkeys(
        [
            *["AutoLand", "AutoShutOff", "ClearMemory", "Dispense", "Error", "Extended"],
            *["ForceWrite", "Harvest", "Idle", "Lock", "NetworkFault", "On", "Open", "Plant"],
            *["Power", "Reset"],
        ],
        "bool",
    ),
    **dict.fromkeys(
        [
            *["Activate", "BestContactFilter", "Bpm", "ChartedNavPoints", "ContactSlotIndex"],
            *["CurrentCode", "CurrentNodeType", "DestinationCode", "DispenseSlot"],
            *["ElevatorLevel", "EntityState", "ExportCount", "FlightControlRule", "ImportCount"],
            *["Index", "LineNumber", "MinedQuantity", "Mode", "NavPoints", "Output", "SignalID"],
            *["Sites", "SoundAlert", "StackSize", "TargetNodeType", "TargetPadIndex"],
            *["TargetSlotIndex", "TotalQuantity"],
        ],
        "int",
    ),
    # Animator.StringToHash of a name, the same number as HASH("name")
    **dict.fromkeys(
        [
            *["CelestialHash", "CelestialParentHash", "ContactTypeId", "NextWeatherHash"],
            *["RecipeHash", "RequestHash", "TargetPrefabHash"],
        ],
        "str",
    ),
    **dict.fromkeys(
        [
            *["Acceleration", "AlignmentError", "Altitude", "Apex", "BurnTimeRemaining"],
            *["Charge", "Chart", "CombustionLimiter", "CompletionRatio", "Density"],
            *["DerivativeGain", "Discover", "DistanceAu", "DistanceKm", "DrillCondition"],
            *["DryMass", "Eccentricity", "ElevatorSpeed", "EnergyConvected", "EnergyRadiated"],
            *["EnvironmentEfficiency", "Gravity", "HealthDamage", "Horizontal"],
            *["HorizontalRatio", "Inclination", "IntegralGain", "InterrogationProgress", "Mass"],
            *["Maximum", "Minimum", "MinimumWattsToContact", "NextWeatherEventTime"],
            *["OperationalTemperatureEfficiency", "OrbitPeriod", "PassedMoles", "PositionX"],
            *["PositionY", "PositionZ", "PowerActual", "PowerGeneration", "PowerPotential"],
            *["PowerRequired", "PressureEfficiency", "PressureExternal", "PressureInternal"],
            *["Progress", "ProportionalGain", "Quantity", "Ratio", "ReEntryAltitude"],
            *["Reagents", "RequiredPower", "Richness", "Rpm", "SemiMajorAxis", "Setpoint"],
            *["SignalStrength", "Size", "SizeX", "SizeZ", "SolarAngle", "SolarIrradiance"],
            *["Stress", "StunDamage", "Survey", "TemperatureDifferentialEfficiency", "Throttle"],
            *["Thrust", "ThrustToWeight", "Time", "TimeToDestination", "TrueAnomaly"],
            *["VelocityRelativeY", "Vertical", "VerticalRatio", "Volume", "VolumeOfLiquid"],
            *["WattsReachingContact", "Weight", "WorkingGasEfficiency"],
        ],
        "float",
    ),
}

# Means something different on each device (a LogicMemory holds hashes, a LogicDial an int,
# a pump a pressure); typed only where a mixin pins the meaning down (_NumericalSetting).
UNTYPED_FIELDS = {"Setting", "SettingInput", "SettingOutput"}

# (prefab name, field) whose values differ from FIELD_TYPES
DEVICE_FIELD_TYPES = {
    # -1 when the rocket has no crew module
    ("StructureRocketAvionics", "Open"): "int",
}


# What reading the LogicSlotType returns (ls/lbs/lbns), read off the decompiled game like
# FIELD_TYPES: Device.GetLogicValue(LogicSlotType, int) and its overrides. Mature and Seeding are
# -1 with no plant, and Growth is the plant's stage. On, Open and Lock writes go to an
# interactable state that reads back as 1 only when it is exactly 1, so they take bool.
SLOT_TYPES: dict[str, str] = {
    **dict.fromkeys(["Lock", "Occupied", "On", "Open"], "bool"),
    **dict.fromkeys(
        [
            *["Class", "FilterType", "FreeSlots", "Growth", "LineNumber", "Mature", "Mode"],
            *["ReferenceId", "Seeding", "SortingClass", "TotalSlots"],
        ],
        "int",
    ),
    **dict.fromkeys(["HarvestedHash", "OccupantHash", "PrefabHash"], "str"),
    **dict.fromkeys(
        [
            *["Charge", "ChargeRatio", "Damage", "Efficiency", "Health", "MaturityRatio"],
            *["MaxQuantity", "Pressure", "PressureAir", "PressureWaste", "Quantity"],
            *["SeedingRatio", "Temperature", "Volume"],
        ],
        "float",
    ),
}


class SlotDump(TypedDict):
    name: str
    type: str


class MemoryDump(TypedDict):
    # 0 when the size depends on the instance (a circuit housing's chip)
    size: int
    access: str
    instructions: str | None


class PrefabDump(TypedDict):
    prefab_name: str
    prefab_hash: int
    hidden: bool
    name: str
    description: str
    device: bool
    logic: dict[str, str]
    slot_logic: dict[str, list[int]]
    slot_logic_write: dict[str, list[int]]
    slots: list[SlotDump]
    memory: MemoryDump | None


class InstructionDump(TypedDict):
    name: str
    opcode: int
    format: str


class GameDump(TypedDict):
    game_version: str
    enums: dict[str, dict[str, int]]
    # LogicType / LogicSlotType -> description
    docs: dict[str, dict[str, str]]
    instruction_sets: dict[str, list[InstructionDump]]
    prefabs: list[PrefabDump]


def _plain(text: str) -> str:
    """Stationpedia text without its TextMeshPro rich-text tags (what the game strips too)."""
    return re.sub(r"<.*?>", "", text)


def _game_text(text: str) -> str:
    """Stationpedia text as plain text; "" for none."""
    text = _plain(text).strip()
    # the game's placeholder for a LogicType or LogicSlotType without a description
    return "" if text == "No description available" else text


def _docstring(text: str, indent: str, *, verbatim: bool = False) -> str:
    """
    `text` as a docstring statement indented by `indent`; "" for no text. Lines are wrapped,
    unless `verbatim` (a table).
    """
    lines: list[str] = []
    for line in text.strip().splitlines():
        line = line.rstrip() if verbatim else " ".join(line.split())
        if not line:
            if lines and lines[-1]:
                lines.append("")
            continue
        lines += [line] if verbatim else textwrap.wrap(line, 100 - len(indent), break_on_hyphens=False)
    if not lines:
        return ""
    body = "\n".join(lines).replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    # a closing quote right after a " would end the string early
    if len(lines) == 1 and len(indent) + len(body) + 6 <= 100 and not body.endswith('"'):
        return f'{indent}"""{body}"""\n'
    return f'{indent}"""\n{textwrap.indent(body, indent)}\n{indent}"""\n'


@dataclass(frozen=True, order=True)
class Property:
    name: str
    can_write: bool = False
    can_read: bool = True

    def generate_code(self, field_types: dict[str, str], docs: dict[str, str]) -> str:
        typ = field_types.get(self.name)
        if self.can_write:
            if typ:
                code = f"    {self.name}: _FieldDescW[{typ}] = _mk_field({typ})\n"
            else:
                code = f"    {self.name}: _FieldDescW = _mk_field()\n"
        else:
            assert self.can_read
            if typ:
                code = f"    {self.name}: _FieldDesc[{typ}] = _mk_field_ro({typ})\n"
            else:
                code = f"    {self.name}: _FieldDesc = _mk_field_ro()\n"
        return code + _docstring(docs[self.name], "    ")


@dataclass(frozen=True)
class Mixin:
    name: str
    properties: frozenset[Property]
    extra: str
    # overrides field_types for this mixin's properties
    types: dict[str, str]

    def generate_code(self, field_types: dict[str, str], docs: dict[str, str]) -> str:
        code = f"class {self.name}:\n"
        for prop in sorted(self.properties):
            code += prop.generate_code(field_types | self.types, docs)
        return code + self.extra


@dataclass(frozen=True)
class SlotClass:
    """The typed slot of the devices whose slots have these LogicSlotTypes (a union over slots)."""

    name: str
    base: str
    # (LogicSlotType, writable)
    fields: tuple[tuple[str, bool], ...]

    def generate_code(self, docs: dict[str, str]) -> str:
        code = f"class {self.name}({self.base}):\n"
        for name, writable in self.fields:
            typ = SLOT_TYPES[name]
            if writable:
                code += f"    {name}: _SlotFieldDescW[{typ}] = _mk_slot_field({typ})\n"
            else:
                code += f"    {name}: _SlotFieldDesc[{typ}] = _mk_slot_field_ro({typ})\n"
            code += _docstring(docs[name], "    ")
        return code


@dataclass(frozen=True)
class InstructionField:
    name: str
    pos: int
    bits: int
    signed: bool
    # bool, int, or str for a hash
    typ: str
    # the bits above the field copy its sign (signed) or are 0 (unsigned) in every word the
    # game writes, and in those sicc packs; decoding is then a shift
    whole: bool = False


@dataclass(frozen=True)
class Instruction:
    name: str
    opcode: int
    # the game's bit layout table, plain text
    table: str
    fields: tuple[InstructionField, ...]
    # where the game's code differs from its table
    note: str = ""


class _InstructionFix(NamedTuple):
    note: str
    # the fields as the game packs them; None: as in the table
    fields: tuple[InstructionField, ...] | None = None
    # the game writes the topmost field, a signed one, zero-extended
    top_zero_extended: bool = False


# (instruction set, instruction) -> how the game's code differs from its layout table, read off
# the decompiled game like FIELD_TYPES. The game sign-extends the topmost field of the words it
# writes when that field is signed, the hashes that tables call unsigned included; only listed
# instructions differ.
_INSTRUCTION_FIXES: dict[tuple[str, str], _InstructionFix] = {
    # RocketCelestialTracker.OnLogicTick: LogicStack.PackByteInt16(1, index, horizontal, vertical)
    ("CelestialTracking", "BodyOrientation"): _InstructionFix(
        "The tracker writes VERTICAL_DECI_DEGREES in bits 8-23, HORIZONTAL_DECI_DEGREES in 24-39"
        " and CELESTIAL_INDEX in 40-47, not as the table shows. A negative horizontal angle"
        " sets CELESTIAL_INDEX to 255. The angles are deci-degrees modulo 360.",
        (
            InstructionField("vertical_deci_degrees", 8, 16, True, "int"),
            InstructionField("horizontal_deci_degrees", 24, 16, True, "int"),
            InstructionField("celestial_index", 40, 8, False, "int"),
        ),
    ),
    # SimpleFabricatorBase / RocketAvionicsDevice: new StackPointerStackAddress(63, 0.0) fixes
    # the opcode it packs at 0; the device keeps the pointer itself
    **dict.fromkeys(
        [("PrinterInstruction", "StackPointer"), ("RocketAvionicsInstruction", "StackPointer")],
        _InstructionFix(
            "The device writes this word with OP_CODE 0, and ignores writes to address 63."
        ),
    ),
    # Silo: LogicStack.PackOpcodeUInt13Int32 shifts the hash in as a uint
    ("SiloInstruction", "SlotContents"): _InstructionFix(
        "The silo writes PREFAB_HASH unsigned; prefab_hash() signs it.", top_zero_extended=True
    ),
}


def _with_whole_top(fields: tuple[InstructionField, ...], top_zero_extended: bool):
    """Mark the topmost field `whole` where the words' writers leave nothing else above it."""
    if not fields:
        return fields
    top = max(fields, key=lambda f: f.pos)
    if top.signed:
        whole = not top_zero_extended
    else:
        # a signed field below sign-extends over it in the game's words (BodyOrientation)
        whole = not any(f.signed for f in fields if f is not top)
    return tuple(dataclasses.replace(f, whole=whole) if f is top else f for f in fields)


_LAYOUT_ROW = re.compile(r"\| (\d+)-(\d+) +\| (\w+) +\| (\w+) +\|")
_LAYOUT_NOTES = [
    re.compile(r"\| VALID ONLY AT ADDRESS \d+ +\|"),
    re.compile(r"\| VALID ONLY AT ADDRESSES \d+ TO \d+ +\|"),
    re.compile(r"\| HASH REPORTED UNSIGNED - MUST BE MANUALLY SIGNED +\|"),
]


def _parse_instruction(dump: InstructionDump) -> Instruction:
    """Parse a bit layout table of LogicStack.FormatInstruction."""
    table = _plain(dump["format"])
    fields: list[InstructionField] = []
    end = 0
    for line in table.splitlines():
        if any(note.fullmatch(line) for note in _LAYOUT_NOTES):
            continue
        m = _LAYOUT_ROW.fullmatch(line)
        assert m, (dump["name"], line)
        lo, hi, name, typ = int(m[1]), int(m[2]), m[3], m[4]
        assert lo == end, (dump["name"], line)
        end = hi + 1
        if name == "UNUSED":
            assert typ == str(hi - lo + 1), (dump["name"], line)
            continue
        type_name, size = typ.rsplit("_", 1)
        assert int(size) == hi - lo + 1, (dump["name"], line)
        if name == "OP_CODE":
            assert (lo, type_name) == (0, "BYTE"), (dump["name"], line)
            continue
        # a hash is signed even where the game writes it unsigned ("MUST BE MANUALLY SIGNED")
        is_hash = "HASH" in name.split("_")
        signed = type_name in ("SBYTE", "SHORT", "INT") or is_hash
        assert type_name in ("BOOL", "BYTE", "SBYTE", "USHORT", "SHORT", "UINT", "INT", "UINT13")
        typ = "bool" if type_name == "BOOL" else "str" if is_hash else "int"
        fields.append(InstructionField(name.lower(), lo, hi - lo + 1, signed, typ))
    assert end == 64, (dump["name"], table)
    return Instruction(dump["name"], dump["opcode"], table, tuple(fields))


@dataclass(frozen=True)
class InstructionSet:
    name: str
    instructions: tuple[Instruction, ...]
    # display names of the devices that have it
    devices: tuple[str, ...]

    def generate_code(self) -> str:
        doc = (
            f"The device instructions {self.name}, in the memory of: {', '.join(self.devices)}."
            f"\n\nA memory word holds one instruction, OP_CODE in bits 0-7 (the IC10 constant"
            f" {self.name}.<name>). `<name>.pack(...)` builds a word, `<name>.<field>(word)`"
            " reads a field of one, and `opcode(word)` reads OP_CODE."
        )
        code = f"class {self.name}:\n" + _docstring(doc, "    ")
        code += (
            "\n    @staticmethod\n"
            "    def opcode(word: _Int) -> _VarRead[int]:\n"
            '        """OP_CODE, bits 0-7."""\n'
            "        return _opcode(word)\n"
        )
        for instr in self.instructions:
            code += f"\n    class {instr.name}:\n"
            note = f"\n\n{textwrap.fill(instr.note, 92)}" if instr.note else ""
            code += _docstring(
                f"{self.name}.{instr.name}\n\n{instr.table}{note}", "        ", verbatim=True
            )
            code += f"\n        OP_CODE = {instr.opcode}\n"
            params = ", ".join(f"{f.name}: _{f.typ.capitalize()}" for f in instr.fields)
            args = "".join(f", ({f.name}, {f.pos}, {f.bits}, {f.signed})" for f in instr.fields)
            code += (
                "\n        @staticmethod\n"
                f"        def pack({params}) -> _Int:\n"
                f"            return _pack({instr.opcode}{args})\n"
            )
            for f in instr.fields:
                doc = f"{f.name.upper()}, bits {f.pos}-{f.pos + f.bits - 1}"
                code += (
                    "\n        @staticmethod\n"
                    f"        def {f.name}(word: _Int) -> _VarRead[{f.typ}]:\n"
                    f'            """{doc}."""\n'
                    f"            return _unpack(word, {f.pos}, {f.bits}, {f.signed}, {f.typ}"
                    + (", whole=True)\n" if f.whole else ")\n")
                )
        return code


@dataclass(frozen=True)
class DeviceClass:
    name: str
    prefab_name: str
    prefab_hash: int
    hidden: bool
    display_name: str
    description: str
    properties: frozenset[Property]
    bases: list[str]
    slots: list[SlotDump]
    slot_cls: SlotClass | None
    # LogicSlotType -> the slots that have it, for those that not every slot has
    slot_only: dict[str, tuple[int, ...]]
    slot_write: dict[str, tuple[int, ...]]
    memory: MemoryDump | None

    def _doc(self) -> str:
        doc = f"{self.display_name} ({self.prefab_name})"
        if self.description:
            doc += f"\n\n{self.description}"
        if self.slot_cls is not None:
            # [first, last, label] of each run of slots with the same label
            runs: list[list[int | str]] = []
            for i, slot in enumerate(self.slots):
                label = slot["name"] or slot["type"]
                if runs and runs[-1][2] == label:
                    runs[-1][1] = i
                else:
                    runs.append([i, i, label])
            doc += "\n\nSlots: " + ", ".join(
                f"{a} {label}" if a == b else f"{a}-{b} {label}" for a, b, label in runs
            )
            doc += "."
        return doc

    def generate_code(self, field_types: dict[str, str], docs: dict[str, str]) -> str:
        code = f"class {self.name}({', '.join(['_DeviceTyped', *self.bases])},):\n"
        code += _docstring(self._doc(), "    ")
        if self.hidden:
            code += "    # hidden in the Stationpedia; may be deprecated or not buildable\n"
        code += f"    _hash: int = {self.prefab_hash}\n"
        if self.prefab_name != f"Structure{self.name}":
            code += f"    _prefab_name = {self.prefab_name!r}\n"
        if self.slot_cls is not None:
            code += f"    _slot_count = {len(self.slots)}\n"
            if self.slot_only:
                code += f"    _slot_only = {self.slot_only!r}\n"
            if self.slot_write:
                code += f"    _slot_write = {self.slot_write!r}\n"
        own_types = {f: t for (p, f), t in DEVICE_FIELD_TYPES.items() if p == self.prefab_name}
        for prop in sorted(self.properties):
            code += prop.generate_code(field_types | own_types, docs)
        if self.slot_cls is not None:
            code += (
                "\n    @property\n"
                f"    def slots(self) -> _Slots[{self.slot_cls.name}]:\n"
                f"        return _Slots(self, {self.slot_cls.name})\n"
            )
        if self.memory is not None:
            code += self._memory_code()
        return code

    def _memory_code(self) -> str:
        assert self.memory is not None
        access = self.memory["access"]
        # no device in 0.2.6429 has write-only memory (sicc._stationeers.MemoryWrite)
        cls = {"R": "_MemoryRead", "RW": "_MemoryReadWrite"}[access]
        ops = {
            "R": "`memory[address]`: get on a pin, getd by ReferenceId otherwise",
            "RW": "`memory[address]`, `memory[address] = value`, `memory.clear()`: get, put and"
            " clr on a pin, getd, putd and clrd by ReferenceId otherwise",
        }[access]
        size = self.memory["size"]
        instructions = self.memory["instructions"]
        # an instruction word is integral; a circuit housing's memory holds whatever its
        # program stores
        typ = "int" if instructions else "_Any"
        doc = f"{size} addresses" if size else "As many addresses as the chip's stack"
        if instructions:
            doc += f", each holding an instruction of {instructions}"
        doc += f".\n\n{ops}."
        code = (
            "\n    @property\n"
            f"    def memory(self) -> {cls}[{typ}]:\n"
            + _docstring(doc, "        ")
            + f"        return {cls}(self, {'int' if instructions else '_AnyType'}, {size or None})\n"
        )
        if instructions:
            code += f"\n    instructions = {instructions}\n"
            code += _docstring("The instructions that this device runs from its memory.", "    ")
        return code


_HEADER = """
# This file is generated by {script} from Stationeers {game_version}.
# do NOT modify

from typing import Any as _Any

from ._api import Bool as _Bool
from ._api import Int as _Int
from ._api import Str as _Str
from ._api import VarRead as _VarRead
from ._core import AnyType as _AnyType
from ._diagnostic import register_exclusion as _register_exclusion
from ._stationeers import Color as _Color
from ._stationeers import DeviceBase
from ._stationeers import DeviceTyped as _DeviceTyped
from ._stationeers import FieldDesc as _FieldDesc
from ._stationeers import FieldDescW as _FieldDescW
from ._stationeers import GasPort
from ._stationeers import MemoryRead as _MemoryRead
from ._stationeers import MemoryReadWrite as _MemoryReadWrite
from ._stationeers import Slot as _SlotBase
from ._stationeers import SlotFieldDesc as _SlotFieldDesc
from ._stationeers import SlotFieldDescW as _SlotFieldDescW
from ._stationeers import Slots as _Slots
from ._stationeers import instr_opcode as _opcode
from ._stationeers import instr_pack as _pack
from ._stationeers import instr_unpack as _unpack
from ._stationeers import mk_field as _mk_field
from ._stationeers import mk_field_ro as _mk_field_ro
from ._stationeers import mk_slot_field as _mk_slot_field
from ._stationeers import mk_slot_field_ro as _mk_slot_field_ro

# debug info points at the user's code, not at the generated helpers
_register_exclusion(__file__)

"""

_GAS_PROP_TEMPLATE = """
    @property
    def {prop_name}(self) -> GasPort:
        assert isinstance(self, DeviceBase)
        return GasPort({args})
"""


def _port_fields(logic_types: dict[str, int]) -> list[str]:
    """LogicTypes that come in all of the per-port variants (Pressure, PressureInput, ...)."""
    return [
        name
        for name in logic_types
        if all(name + suf in logic_types for suf in PORT_SUFFIXES[1:])
        # SettingInput / SettingOutput exist, but are not a gas-port reading
        and name != "Setting"
    ]


def _check_hand_written(dump: GameDump, port_fields: list[str]) -> None:
    color_game = dump["enums"]["ColorType"]
    color_sicc = {c.name: c.value for c in Color}
    if color_game != color_sicc:
        raise RuntimeError(f"sicc._stationeers.Color is out of date; game ColorType is {color_game}")

    gas_port_sicc = {f.name for f in dataclasses.fields(GasPort)} - {"device", "suffix"}
    if gas_port_sicc != set(port_fields):
        raise RuntimeError(
            "sicc._stationeers.GasPort fields are out of date:\n"
            f"  missing: {sorted(set(port_fields) - gas_port_sicc)}\n"
            f"  extra:   {sorted(gas_port_sicc - set(port_fields))}"
        )


def _field_types(port_fields: list[str]) -> dict[str, str]:
    port_types = {
        name + suf: "bool" if name == "Combustion" else "float"
        for name in port_fields
        for suf in PORT_SUFFIXES
    }
    assert not port_types.keys() & FIELD_TYPES.keys(), port_types.keys() & FIELD_TYPES.keys()
    return FIELD_TYPES | port_types


def generate(dump: GameDump) -> str:
    port_fields = _port_fields(dump["enums"]["LogicType"])
    _check_hand_written(dump, port_fields)
    field_types = _field_types(port_fields)

    def mixin(
        name: str, *properties: Property, extra: str = "", types: dict[str, str] | None = None
    ) -> Mixin:
        return Mixin(name, frozenset(properties), extra, types or {})

    P = Property
    mixins: list[Mixin] = []
    for suf in PORT_SUFFIXES:
        args = [
            "device=self",
            f"suffix={suf!r}",
            *(f"{name}=self.{name}{suf}" for name in port_fields),
        ]
        extra = _GAS_PROP_TEMPLATE.format(
            prop_name=suf.lower() if suf else "gas_info", args=",".join(args)
        )
        mixins.append(mixin(f"_BaseGas{suf}", *(P(name + suf) for name in port_fields), extra=extra))
    mixins += [
        mixin("_Powered", P("Power"), P("RequiredPower"), P("On", True)),
        # in 0.2.6429, only gas devices (Setting = DeviceAtmospherics.OutputSetting) and
        # transformers have all three
        mixin(
            "_NumericalSetting",
            P("Maximum"),
            P("Ratio"),
            P("Setting", True),
            types={"Setting": "float"},
        ),
        mixin("_ImportExport", P("ImportCount"), P("ExportCount"), P("ClearMemory", True, False)),
        mixin("_ImportOnly", P("ImportCount"), P("ClearMemory", True, False)),
        mixin("_PowerMeasure", P("PowerActual"), P("PowerPotential")),
        mixin("_Volume", P("Volume"), P("VolumeOfLiquid")),
        mixin("_Battery", P("Charge"), P("Maximum")),
        mixin("_Door", P("Idle"), P("Lock", True), P("Mode", True), P("Open", True), P("Setting", True)),
        mixin(
            "_Machine",
            P("Activate", True),
            P("CompletionRatio"),
            P("Error"),
            P("Lock", True),
            P("Open", True),
            P("Reagents"),
            P("RecipeHash", True),
        ),
        mixin(
            "_Sleeper",
            P("Activate", True),
            P("EntityState"),
            P("Error"),
            P("Lock", True),
            P("Open", True),
        ),
    ]

    device_prefabs = [p for p in dump["prefabs"] if p["device"]]
    unknown_slot_types = {t for p in device_prefabs for t in p["slot_logic"]} - SLOT_TYPES.keys()
    if unknown_slot_types:
        raise RuntimeError(
            f"LogicSlotTypes {sorted(unknown_slot_types)} are not typed; check what the game"
            " returns and add them to SLOT_TYPES"
        )
    # every device's slots have these; the other LogicSlotTypes go to subclasses of _Slot
    slot_unions = [frozenset(p["slot_logic"]) for p in device_prefabs if p["slot_logic"]]
    slot_base = slot_unions[0].intersection(*slot_unions[1:])
    slot_classes = {"_Slot": SlotClass("_Slot", "_SlotBase", tuple((t, False) for t in sorted(slot_base)))}

    def slot_class(prefab: PrefabDump) -> SlotClass:
        extras = sorted(prefab["slot_logic"].keys() - slot_base)
        writable = prefab["slot_logic_write"].keys()
        assert writable <= prefab["slot_logic"].keys() and not writable & slot_base, prefab
        if not extras:
            return slot_classes["_Slot"]
        cls = SlotClass("_Slot" + "".join(extras), "_Slot", tuple((t, t in writable) for t in extras))
        # devices whose slots have the same LogicSlotTypes share a class, named after them
        assert slot_classes.setdefault(cls.name, cls) == cls, (cls, slot_classes[cls.name])
        return cls

    instruction_sets = {
        name: tuple(_parse_instruction(i) for i in instructions)
        for name, instructions in dump["instruction_sets"].items()
    }
    for (set_name, instr_name), fix in _INSTRUCTION_FIXES.items():
        instructions = instruction_sets[set_name]
        (i,) = (k for k, instr in enumerate(instructions) if instr.name == instr_name)
        instr = instructions[i]
        if fix.fields is not None:
            # the same fields as the table, placed differently; a changed table needs a new look
            assert {f.name for f in fix.fields} == {f.name for f in instr.fields}, (instr, fix)
        fixed = dataclasses.replace(instr, note=fix.note, fields=fix.fields or instr.fields)
        instruction_sets[set_name] = instructions[:i] + (fixed,) + instructions[i + 1 :]
    instruction_sets = {
        set_name: tuple(
            dataclasses.replace(
                instr,
                fields=_with_whole_top(
                    instr.fields,
                    _INSTRUCTION_FIXES.get((set_name, instr.name), _InstructionFix("")).top_zero_extended,
                ),
            )
            for instr in instructions
        )
        for set_name, instructions in instruction_sets.items()
    }

    devices: dict[str, DeviceClass] = {}
    for prefab in device_prefabs:
        # hidden devices are kept: deprecated ones still work in existing saves
        prefab_name = prefab["prefab_name"]
        assert crc32(prefab_name) == prefab["prefab_hash"], prefab_name

        properties = frozenset(
            Property(name, can_write="W" in access, can_read="R" in access)
            for name, access in prefab["logic"].items()
            if name not in DEVICE_FIELDS
        )
        if not properties and not prefab["slot_logic"] and not prefab["memory"]:
            continue

        bases: list[str] = []
        for m in mixins:
            if m.properties <= properties:
                properties -= m.properties
                bases.append(m.name)

        # "StructureTransformerMedium(Reversed)" -> TransformerMediumReversed
        name = re.sub(r"\W", "", prefab_name.removeprefix("Structure"))
        assert name.isidentifier() and name not in devices, prefab_name
        all_slots = list(range(len(prefab["slots"])))
        devices[name] = DeviceClass(
            name,
            prefab_name,
            prefab["prefab_hash"],
            prefab["hidden"],
            prefab["name"],
            _game_text(prefab["description"]),
            properties,
            sorted(bases),
            prefab["slots"],
            slot_class(prefab) if prefab["slot_logic"] else None,
            {t: tuple(idx) for t, idx in prefab["slot_logic"].items() if idx != all_slots},
            {t: tuple(idx) for t, idx in prefab["slot_logic_write"].items()},
            prefab["memory"],
        )

    for prefab_name, field in DEVICE_FIELD_TYPES:
        (dev,) = (d for d in devices.values() if d.prefab_name == prefab_name)
        # a mixin would type the field for every device sharing it
        assert field in {p.name for p in dev.properties}, (prefab_name, field)

    docs = {kind: {k: _game_text(v) for k, v in d.items()} for kind, d in dump["docs"].items()}
    code = _HEADER.format(script=Path(__file__).name, game_version=dump["game_version"])
    for name in sorted(slot_classes, key=lambda n: (n != "_Slot", n)):
        code += slot_classes[name].generate_code(docs["LogicSlotType"])
    for name, instructions in instruction_sets.items():
        users = sorted(
            {d.display_name for d in devices.values() if d.memory and d.memory["instructions"] == name}
        )
        code += InstructionSet(name, instructions, tuple(users)).generate_code()
    for m in mixins:
        code += m.generate_code(field_types, docs["LogicType"])
    for name in sorted(devices):
        code += devices[name].generate_code(field_types, docs["LogicType"])
    return code


def main() -> None:
    dump: GameDump = json.loads(DUMP_FILE.read_text(encoding="utf-8"))
    OUT_FILE.write_text(generate(dump), encoding="utf-8")
    subprocess.run(["black", "-q", str(OUT_FILE)], check=True)
    print("Output written to", OUT_FILE)

    used = {name for p in dump["prefabs"] if p["device"] for name in p["logic"]}
    typed = _field_types(_port_fields(dump["enums"]["LogicType"])).keys()
    unclassified = sorted(used - typed - UNTYPED_FIELDS - DEVICE_FIELDS)
    if unclassified:
        print(
            "warning: fields left untyped (Any); check what the game returns and add them to"
            f" FIELD_TYPES: {', '.join(unclassified)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
