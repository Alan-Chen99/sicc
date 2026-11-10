"""
Satellite dish signal locator and interrogator.

Based on the "Dish Control" script of https://stationeers-wiki.com/Satellite_Tracking,
section "Shuttle Cargoes Auto Filtration/Interrogation and Call Down".

The target signal ID is read from the LED display named "LED_LOCKED_SIGID", which the program
`filtration` of `dish_trader.py` sets to the signal ID of a trader with matching cargo.

- ID is 0: roam, sweeping DISH_M horizontally 2 degrees per tick.
- ID is set: the scan dish finds both horizontal edges of the signal, then both
  vertical edges, and both dishes are aimed at the middle. The comm dish then
  interrogates the signal; on completion the transmitter sounds an alert and the
  LED is cleared. If the scan dish loses the signal during the interrogation, the
  LED is cleared too.
- Changing the ID abandons the current search.

The small dish (if configured) scans when the LED named "LED_TIER" reads below 2;
the large dish (if configured) interrogates. Both are off outside a search.

Every device is addressed by prefab type and name over the data network, so the IC
housing needs no pins set.
"""

from typing import Callable

from sicc import *
from sicc.devices import *

################################################################################
# Configuration

DISH_M = SatelliteDish("Medium Satellite Dish 1")
# Optional extra dishes, e.g. SmallSatelliteDish("NAME") / LargeSatelliteDish("NAME").
# A dish left as None adds no lines to the output.
DISH_S: SmallSatelliteDish | None = None
DISH_L: LargeSatelliteDish | None = None

# Input: the signal ID to locate, 0 for none. This program clears it when an
# interrogation completes or the scan dish loses the signal during one.
LED_TARGET = ConsoleLED5("LED_LOCKED_SIGID")
# Input: picks the scan dish (see pick_dishes).
LED_TIER = ConsoleLED5("LED_TIER")
# Plays sound alert 37 when an interrogation completes.
TRANSMITTER = Device("StructureLogicTransmitter", "Hardsuit LT 1")

################################################################################

# Any dish size; they share the logic types used here.
Dish = SatelliteDish | SmallSatelliteDish | LargeSatelliteDish

# The signal ID being searched for; set by main, read by tick() and search().
target = Variable(float)
# Start of the main loop; tick() jumps here to abandon a search.
RESTART = label_ref("restart")


@transpile
def tick():
    """
    Wait one tick. If the LED no longer shows `target`, the search is over: jump to
    RESTART instead of returning.

    A subroutine (not inlined), so each call costs one line.
    """
    yield_()
    if LED_TARGET.Setting != target:
        jump(RESTART)


@transpile(inline=True)
def tick_until(done: Static[Callable[[], Bool]]):
    """Tick at least once, then until `done()` holds."""
    while True:
        tick()
        if done():
            break


@transpile(inline=True)
def settle(dish: Dish):
    """Tick until `dish` reports Idle; called after aiming it."""
    tick_until(lambda: dish.Idle != 0)


@transpile(inline=True)
def sweep_horizontal(scan: Dish, step: Static[int]) -> Float:
    """
    Turn `scan` `step` degrees per tick while it sees the signal.
    Returns the last angle at which it saw the signal.
    """
    seen_at = scan.Horizontal
    while True:
        tick()
        if scan.SignalID == -1:
            return seen_at
        seen_at = scan.Horizontal
        scan.Horizontal = (seen_at + step) % 360


@transpile(inline=True)
def sweep_vertical(scan: Dish, step: Static[int], at_end: Static[Callable[[Float], Bool]]) -> Float:
    """
    Tilt `scan` `step` degrees per tick while it sees the signal, until `at_end(angle)`.
    Returns the last angle at which it saw the signal.
    """
    seen_at = scan.Vertical
    while True:
        tick()
        if scan.SignalID == -1:
            break
        seen_at = scan.Vertical
        if at_end(seen_at):
            break
        scan.Vertical = seen_at + step
    return seen_at


@transpile(inline=True)
def roam() -> Float:
    """Tilt DISH_M to 60 and turn it 2 degrees per tick until a target is set; return it."""
    DISH_M.Vertical = 60
    DISH_M.BestContactFilter = -1
    DISH_M.Setting = 500
    DISH_M.On = True
    while True:
        yield_()
        led = LED_TARGET.Setting
        if led != 0:
            return led
        DISH_M.Horizontal = (DISH_M.Horizontal + 2) % 360


def pick_dishes() -> tuple[Dish, Dish]:
    """
    Return the (scan, comm) dishes: DISH_S scans when LED_TIER < 2, DISH_L interrogates.

    Plain python: @transpile does not allow reassigning the trace-time `scan` and `comm` under
    an `if`, even one on a python bool such as `DISH_S is not None`.
    """
    scan = DISH_M.as_generic_device_keep_type()
    comm = DISH_M.as_generic_device_keep_type()
    if DISH_S is not None:
        use_small = Variable(LED_TIER.Setting < 2)
        scan = select(use_small, DISH_S.as_generic_device_keep_type(), scan)
        with if_(use_small):
            DISH_S.Setting = 200
    if DISH_L is not None:
        comm = DISH_L.as_generic_device_keep_type()
    return scan, comm


@transpile(inline=True)
def search(scan: Dish, comm: Dish):
    """
    Locate signal `target` with `scan`, aim both dishes at it, interrogate it with
    `comm` and clear the LED.
    """
    scan.BestContactFilter = target
    scan.On = True
    tick_until(lambda: scan.SignalID != -1)

    # Horizontal edges: sweep down to one, return to it, sweep up to the other.
    h_low = sweep_horizontal(scan, -5)
    scan.Horizontal = h_low
    settle(scan)
    h_high = sweep_horizontal(scan, 5)
    # the upward sweep can wrap past 360 to a smaller angle
    wrap = 360 if h_high < h_low else 0
    h_mid = (h_low + h_high + wrap) / 2 % 360
    scan.Horizontal = h_mid
    comm.On = True
    comm.Horizontal = h_mid
    settle(scan)

    # Vertical edges: the same, up first; the sweeps stop near 90 and 0.
    v_high = sweep_vertical(scan, 5, lambda v: v >= 89.9)
    scan.Vertical = v_high
    settle(scan)
    v_low = sweep_vertical(scan, -5, lambda v: v <= 0.1)
    v_mid = (v_low + v_high) / 2
    scan.Vertical = v_mid
    comm.Vertical = v_mid
    settle(scan)

    # Interrogate until done or the scan dish loses the signal.
    while True:
        tick()
        if scan.SignalID == -1:
            break
        settle(comm)
        comm.BestContactFilter = target
        comm.Setting = 50000
        comm.Activate = True
        if comm.InterrogationProgress >= 1:
            sleep(2)
            TRANSMITTER["SoundAlert"] = 37
            break

    LED_TARGET.Setting = 0


@program
@transpile
def main():
    while True:
        label(RESTART)
        # On a python bool, `if` picks its branch at trace time: a dish left as None adds no
        # lines.
        if DISH_S is not None:
            DISH_S.On = False
        if DISH_L is not None:
            DISH_L.On = False
        target.value = roam()
        scan, comm = pick_dishes()
        search(scan, comm)


if __name__ == "__main__":
    main.cli()
