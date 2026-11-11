"""
Generate sicc/devices.py from game_data/stationeers_logic.json.

The JSON is dumped from the game by game_data/dump.sh. This script also checks that the
hand-written sicc._stationeers.Color and sicc._stationeers.GasPort match the dumped game enums.

Usage: python scripts/build_types.py
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
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


class PrefabDump(TypedDict):
    prefab_name: str
    prefab_hash: int
    hidden: bool
    device: bool
    logic: dict[str, str]


class GameDump(TypedDict):
    game_version: str
    enums: dict[str, dict[str, int]]
    prefabs: list[PrefabDump]


@dataclass(frozen=True, order=True)
class Property:
    name: str
    can_write: bool = False
    can_read: bool = True

    def generate_code(self, field_types: dict[str, str]) -> str:
        typ = field_types.get(self.name)
        if self.can_write:
            if typ:
                return f"  {self.name}: _FieldDescW[{typ}] = _mk_field({typ})\n"
            return f"  {self.name}: _FieldDescW = _mk_field()\n"
        assert self.can_read
        if typ:
            return f"  {self.name}: _FieldDesc[{typ}] = _mk_field_ro({typ})\n"
        return f"  {self.name}: _FieldDesc = _mk_field_ro()\n"


@dataclass(frozen=True)
class Mixin:
    name: str
    properties: frozenset[Property]
    extra: str
    # overrides field_types for this mixin's properties
    types: dict[str, str]

    def generate_code(self, field_types: dict[str, str]) -> str:
        code = f"class {self.name}:\n"
        for prop in sorted(self.properties):
            code += prop.generate_code(field_types | self.types)
        return code + self.extra


@dataclass(frozen=True)
class DeviceClass:
    name: str
    prefab_name: str
    prefab_hash: int
    hidden: bool
    properties: frozenset[Property]
    bases: list[str]

    def generate_code(self, field_types: dict[str, str]) -> str:
        code = f"class {self.name}({', '.join(['_DeviceTyped', *self.bases])},):\n"
        if self.hidden:
            code += "  # hidden in the Stationpedia; may be deprecated or not buildable\n"
        code += f"  _hash: int = {self.prefab_hash}\n"
        if self.prefab_name != f"Structure{self.name}":
            code += f"  _prefab_name = {self.prefab_name!r}\n"
        own_types = {f: t for (p, f), t in DEVICE_FIELD_TYPES.items() if p == self.prefab_name}
        for prop in sorted(self.properties):
            code += prop.generate_code(field_types | own_types)
        return code


_HEADER = """
# This file is generated by {script} from Stationeers {game_version}.
# do NOT modify

from ._stationeers import Color as _Color
from ._stationeers import DeviceBase
from ._stationeers import DeviceTyped as _DeviceTyped
from ._stationeers import FieldDesc as _FieldDesc
from ._stationeers import FieldDescW as _FieldDescW
from ._stationeers import GasPort
from ._stationeers import mk_field as _mk_field
from ._stationeers import mk_field_ro as _mk_field_ro

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

    devices: dict[str, DeviceClass] = {}
    for prefab in dump["prefabs"]:
        # hidden devices are kept: deprecated ones still work in existing saves
        if not prefab["device"]:
            continue
        prefab_name = prefab["prefab_name"]
        assert crc32(prefab_name) == prefab["prefab_hash"], prefab_name

        properties = frozenset(
            Property(name, can_write="W" in access, can_read="R" in access)
            for name, access in prefab["logic"].items()
            if name not in DEVICE_FIELDS
        )
        if not properties:
            continue

        bases: list[str] = []
        for m in mixins:
            if m.properties <= properties:
                properties -= m.properties
                bases.append(m.name)

        # "StructureTransformerMedium(Reversed)" -> TransformerMediumReversed
        name = re.sub(r"\W", "", prefab_name.removeprefix("Structure"))
        assert name.isidentifier() and name not in devices, prefab_name
        devices[name] = DeviceClass(
            name, prefab_name, prefab["prefab_hash"], prefab["hidden"], properties, sorted(bases)
        )

    for prefab_name, field in DEVICE_FIELD_TYPES:
        (dev,) = (d for d in devices.values() if d.prefab_name == prefab_name)
        # a mixin would type the field for every device sharing it
        assert field in {p.name for p in dev.properties}, (prefab_name, field)

    code = _HEADER.format(script=Path(__file__).name, game_version=dump["game_version"])
    for m in mixins:
        code += m.generate_code(field_types)
    for name in sorted(devices):
        code += devices[name].generate_code(field_types)
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
