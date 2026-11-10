"""
Satellite dish signal tracker.

Based on the script at https://pastebin.com/RKtQw3Vd, linked as "Source Code" from
https://stationeers-wiki.com/Satellite_Tracking. The wiki marks that script as outdated since
the trader updates.

The dish scans the sky in two rows, at vertical 80 and 50, turning 20 degrees per step and
reversing direction at each end of a row. When it sees a signal other than the one it tracked
last, the light turns on and the dish homes in on the signal, the light's Setting showing the
signal strength. Once the strength reaches 96% the dish holds still. Turning the light off
ends the tracking at any time; the dish then returns to where it found the signal and
continues the scan.

Wiring:
- `d0`: Satellite Dish
- `d1`: Diode Slide (the light)
"""

from sicc import *
from sicc.devices import *
from sicc.functions import clamp

DISH = SatelliteDish(pin=d0)
LIGHT = DiodeSlide(pin=d1)

ROW_HIGH = 80
ROW_LOW = 50
SCAN_STEP = 20
LOCKED_STRENGTH = 0.96


@transpile
def settle():
    """Tick until the dish has turned to where it was aimed."""
    while True:
        yield_()
        if DISH.Idle != 0:
            break


@transpile(inline=True)
def track():
    """
    Home in on the signal the dish sees until the light is turned off.

    Hill climbing, one axis at a time: step by up to 30 degrees, less the stronger the signal,
    and reverse that axis' direction when the signal does not get stronger. Horizontal wraps
    around; vertical stays within 0-90, where a step that cannot move counts as not stronger.
    """
    LIGHT.On = True
    # A device read has type Any unless given one, and a local of type Any cannot later be
    # assigned a number, such as the 0 or 90 that clamp() below can return.
    h = DISH.Horizontal.get(typ=float)
    v = DISH.Vertical.get(typ=float)
    v_dir = -1
    h_dir = 1
    while True:
        before = DISH.SignalStrength
        v = clamp(v + (1 - before) * 30 * v_dir, 0, 90)
        DISH.Vertical = v
        settle()
        if DISH.SignalStrength <= before:
            v_dir = -v_dir

        before = DISH.SignalStrength
        h = (h + (1 - before) * 30 * h_dir) % 360
        DISH.Horizontal = h
        settle()
        strength = DISH.SignalStrength
        if strength <= before:
            h_dir = -h_dir
        LIGHT.Setting = strength

        while LIGHT.On != 0 and DISH.SignalStrength >= LOCKED_STRENGTH:
            yield_()
        if LIGHT.On == 0:
            break


@program
@transpile
def main():
    LIGHT.On = False
    # the signal tracked last, to skip it when the scan passes it again
    last = 0.0
    h = 0
    v = ROW_HIGH
    step = SCAN_STEP
    while True:
        h += step
        if h % 360 == 0:
            # end of a row: turn around on the other row
            step = -step
            v = ROW_LOW if v == ROW_HIGH else ROW_HIGH
        DISH.Vertical = v
        DISH.Horizontal = h
        settle()

        signal = DISH.SignalID
        # No signal reads 0 in this script's original and -1 in the newer Dish Control script
        # on the same wiki page; signal IDs are positive.
        if signal > 0 and signal != last:
            track()
            last = DISH.SignalID
            DISH.Vertical = v
            DISH.Horizontal = h
            settle()


if __name__ == "__main__":
    main.cli()
