"""
SICC example equivalent to:
https://github.com/SnorreSelmer/stationeers_ic10/blob/45638ddabd9c569eb0a03f3f39495097cbbad66b/old-scripts/auto_adv_furnace_library.ic10

Wiring (pins):
- `d0`: Furnace
- `d1`: On/off switch (reads `Setting`, writes Furnace `On`)
- `d2`: Logic Dial (select recipe #1..17 via `Setting`)
- `d3`: IC Furnace Controller (writes `Setting`)
"""

from __future__ import annotations

from sicc import *
from sicc.devices import *
from sicc.functions import *

host = CircuitHousing(pin=db)

furnace = AdvancedFurnace(pin=d0)
furnace_on = LogicSwitch(pin=d1)
dial = LogicDial(pin=d2)
controller = CircuitHousing(pin=d3)

RES_MAX = 17
DELAY_TICKS = 6


@dataclass
class Recipe:
    ingot_hash: Str
    pres_mm: Float
    temp_mm: Float


def _recipe(
    ingot_name: str, pres_min: float, pres_max: float, temp_min: float, temp_max: float
) -> Recipe:
    """Create a recipe from unpacked min/max values, packing them for storage."""
    pres_mm = pres_min + pres_max / 100000.0
    temp_mm = temp_min + temp_max / 10000.0
    return Recipe(ingot_name, pres_mm, temp_mm)


# Recipe order matches the original IC10 jump table:
# Iron, Copper, Silicon, Silver, Gold, Lead, Nickel, Steel, Electrum, Solder,
# Invar, Constantan, Astroloy, Hastelloy, Inconel, Stellite, Waspaloy
RECIPES: list[Recipe] = [
    _recipe("ItemIronIngot", 1100, 10000, 900, 2000),
    _recipe("ItemCopperIngot", 1100, 10000, 700, 2000),
    _recipe("ItemSiliconIngot", 1100, 10000, 1000, 2000),
    _recipe("ItemSilverIngot", 1100, 10000, 700, 2000),
    _recipe("ItemGoldIngot", 1100, 10000, 700, 2000),
    _recipe("ItemLeadIngot", 1100, 10000, 500, 2000),
    _recipe("ItemNickelIngot", 1100, 10000, 900, 2000),
    _recipe("ItemSteelIngot", 1100, 10000, 1000, 2000),
    _recipe("ItemElectrumIngot", 1000, 2000, 700, 2000),
    _recipe("ItemSolderIngot", 1100, 10000, 400, 500),
    _recipe("ItemInvarIngot", 19000, 19000, 1300, 1400),
    _recipe("ItemConstantanIngot", 21000, 30000, 1100, 2000),
    _recipe("ItemAstroloyIngot", 31000, 39000, 1100, 2000),
    _recipe("ItemHastelloyIngot", 26000, 29000, 975, 975),
    _recipe("ItemInconelIngot", 23750, 23750, 700, 2000),
    _recipe("ItemStelliteIngot", 11000, 19000, 1900, 2100),
    _recipe("ItemWaspaloyIngot", 51000, 51000, 500, 700),
]


@transpile(inline=True)
def _round_to(step: float, x: Float) -> Float:
    """Round x to the nearest multiple of step."""
    return round_(x / step) * step


@transpile(inline=True)
def _decode_mm(mm: Float, *, scale: float) -> tuple[Float, Float]:
    """
    Decode the packed `min.max` convention.

    - min = floor(mm)
    - max = round((mm - min) * scale)
    """
    min_ = floor(mm)
    max_ = round_((mm - min_) * scale)
    return min_, max_


@program
@transpile
def main():
    table = stack_var(RECIPES)

    # Previous recipe index for change detection
    res_nr_mem: Int = 0
    # Delay counter (changes only applied after DELAY_TICKS of stable selection)
    delay_counter: Int = 0

    while True:
        comment("=== Main loop start ===")

        # Read on/off switch and control furnace power
        furnace.On = furnace_on.Setting

        # Read the current recipe selection from dial
        # Dial.Mode is set to RES_MAX so dial ranges 1..RES_MAX
        dial.Mode = RES_MAX
        # Clamp to valid range [1, RES_MAX]
        res_nr = clamp(dial.Setting, 1, RES_MAX)

        # Table is 0-indexed, dial is 1-indexed
        recipe = table[res_nr - 1].read()

        # Store hash in db.Setting for display purposes
        host.Setting = recipe.ingot_hash

        # Open furnace if current recipe matches our target hash
        furnace.Open = furnace.RecipeHash == recipe.ingot_hash

        # Check if recipe changed (compare recipe index)
        recipe_changed = res_nr != res_nr_mem
        res_nr_mem = res_nr

        # Increment delay counter, reset if recipe changed
        delay_counter = 0 if recipe_changed else delay_counter + 1

        # Wait for delay before applying temperature/pressure settings
        if delay_counter < DELAY_TICKS:
            yield_()
            continue

        comment("=== Applying furnace settings ===")

        # Decode packed pressure and temperature values
        pres_min, pres_max = _decode_mm(recipe.pres_mm, scale=100000)
        temp_min, temp_max = _decode_mm(recipe.temp_mm, scale=10000)

        # Read current furnace values and round to steps
        # Temperature rounds to 100K steps, pressure to 500kPa steps
        current_temp = _round_to(100, furnace.Temperature)
        current_pres = _round_to(500, furnace.Pressure)

        # Clamp to target ranges
        set_temp = clamp(current_temp, temp_min, temp_max)
        set_pres = clamp(current_pres, pres_min, pres_max)

        # Combine into single setting value for IC Furnace Controller
        # Format: SetTemp/10000 + SetPres
        controller.Setting = set_temp / 10000 + set_pres

        yield_()


if __name__ == "__main__":
    main.cli()
