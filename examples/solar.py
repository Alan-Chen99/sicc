"""
adapated from `https://stationeers-wiki.com/IC10#Solar_Panel_2-axis_tracking`

Setup notes (from the original script):
- Place all panels in a uniform manner.
- Set one to 15 Vertical (min value), 0 Horizontal; note the direction it faces.
- Place the daylight sensor flat pointing in that direction (cable port facing opposite).
"""

from sicc import *
from sicc.devices import *

# Alias the sensor to d0
sensor = d0

# Define the panel variants
panels = [
    SolarPanel(),
    SolarPanel45(),
    SolarPanelDual(),
    SolarPanelFlat(),
    SolarPanel45Reinforced(),
    SolarPanelDualReinforced(),
    SolarPanelFlatReinforced(),
    SolarPanelReinforced(),
]

# Edit this to face sunrise (matches the original script's inline comments)
RESET_HORIZONTAL = 270
RESET_VERTICAL = 0


@shared_generator  # remove @shared_generator to unroll
def iter_panels():
    for x in panels:
        yield x.as_generic_device_keep_type()


@program
@transpile
def main():
    while True:
        yield_()

        # Check for daylight (sensor.Activate == 0 -> reset)
        if sensor.Activate == 0:
            yield_()

            # Reset panels to sunrise-facing horizontal and 0 vertical
            for p in iter_panels():
                p.Horizontal = RESET_HORIZONTAL
                p.Vertical = RESET_VERTICAL

            sleep(10)
            continue

        # Read the horizontal data and set batch to the panels.
        horizontal = sensor.Horizontal
        # Read vertical data and subtract from 90 (original: `sub r0 90 r0`), then set batch.
        vertical = 90 - sensor.Vertical

        for p in iter_panels():
            p.Horizontal = horizontal
            p.Vertical = vertical


if __name__ == "__main__":
    main.cli()
