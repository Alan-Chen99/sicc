"""
Trader filtration and Dish Control on one IC: the program `filtration` of dish_trader.py and
dish_control.py in one program, on the Medium Satellite Dish DISH_M.

Filtration and Dish Control are two static tasks, so either can run alone or beside other tasks;
main steps each once per tick, the rate dish_control.py runs at. Filtration yields after writing
its request, after every 8 stack addresses it reads back and before clearing the Logic Memories
left over, so each part fits in one tick with Dish Control's step: a round takes 5 ticks.
Filtration hands the signal ID to locate to Dish Control in the variable `target`.

Filtration (see dish_trader.py, which describes the devices to build and label): each round
writes a request for the strongest contact's cargo into the dish's stack, filtered by
SLM_ITEM_FILTER if set, and reads the cargo back onto the Logic Memories named in ITEM_MEMORIES
and GAS_MEMORIES. With a filter set, while Dish Control roams, a contact with matching cargo
becomes the target, unless it was the target last time, whether or not that attempt succeeded.
This check comes before the cargo is shown, so Dish Control turns to the trader a few ticks
sooner than it would after the display.

Dish Control (see dish_control.py): with no target, the dish roams at vertical 60, turning 2
degrees per tick. With a target, the dish settles, finds both horizontal edges of the signal,
then both vertical edges, aims at the middle and interrogates the signal. On completion the
transmitter sounds an alert and the target is cleared; the target is cleared too if the dish
loses the signal during the interrogation, or does not see it once it settles after roaming
(the sweeps then end at once and the dish stays put; dish_control.py waits for the signal, which
only the LED below could cancel).

Dropped to fit in 128 lines:
- the LED LED_LOCKED_SIGID: filtration and Dish Control share `target` instead, so a target can
  no longer be set or cleared by hand
- filtration: the contact's metadata on the LEDs LED_CON_TYPE_ID, LED_TIER, LED_SHUTTLE_TYPE,
  LED_WATT_TO_RESOLVE, LED_CONTACTED and LED_LIFETIME (LED_SIGID still shows the contact)
- filtration: waiting for a dish named as DISH_M to exist; without one, the IC stops with an
  error
- Dish Control: the optional scan and comm dishes (DISH_S, DISH_L), as DISH_M does both; the
  2 s sleep before the alert, which would stall filtration too
- Dish Control: the vertical sweeps go down first, as the horizontal ones do, and stop a step
  short of 0 and 90 rather than at them

Every device is addressed by name over the data network, so the IC housing needs no pins set.
"""

from sicc import *
from sicc import functions as f
from sicc.devices import *

################################################################################
# Configuration

DISH_M = SatelliteDish("Medium Satellite Dish 1")

# Logic Memories that receive the cargo, as in dish_trader.py: the n-th memory (from 0) has
# the name hash ITEM_HASH + n or GAS_HASH + n.
ITEM_HASH = 1_000_000_000
ITEM_MEMORIES = [
    "SLMP1_cF5saO", "SLMP2_h6mrT5", "SLMP3_3cn8Dy", "SLMP4_GkV2jz", "SLMP5_QA601h",
    "SLMP6_X0zc5W", "SLMP7_PWnvdB", "SLMP8_UMFmcA", "SLMP9_sxmyKl", "SLMP10_GZJSzt",
    "SLMP11_fiXLkI", "SLMP12_JcGgKY", "SLMP13_4RdMQl", "SLMP14_Qkvdxm", "SLMP15_s5B5GH",
    "SLMP16_mlbJuG", "SLMP17_FHlZ9o", "SLMP18_u8wXEH", "SLMP19_nRLpJB", "SLMP20_SQSZrP",
]  # fmt: skip
GAS_HASH = 1_000_001_000
GAS_MEMORIES = [
    "SLMG1_UhOxBi", "SLMG2_5ScSpt", "SLMG3_sU2un5", "SLMG4_ngVvsD", "SLMG5_yMwE3O",
    "SLMG6_N0mAD8", "SLMG7_g4xq1u", "SLMG8_TDcsMR", "SLMG9_WeH96W", "SLMG10_mvjGIL",
    "SLMG11_lJKact",
]  # fmt: skip
assert [crc32(name) - ITEM_HASH for name in ITEM_MEMORIES] == list(range(len(ITEM_MEMORIES)))
assert [crc32(name) - GAS_HASH for name in GAS_MEMORIES] == list(range(len(GAS_MEMORIES)))

# Open: list what the trader buys; closed: what it sells.
SELL_BUY_SWITCH = LogicSwitch2("TRADER_SELL_BUY_SWITCH")
LED_SIGID = ConsoleLED5("LED_SIGID")
# The filter: a prefab hash, or gas bitflags when FILTER_GAS_SWITCH is open. 0 for none.
ITEM_FILTER = LogicMemory("SLM_ITEM_FILTER")
FILTER_GAS_SWITCH = LogicSwitch2("FILTER_GAS_SWITCH")
# Plays sound alert 37 when an interrogation completes.
TRANSMITTER = Device("StructureLogicTransmitter", "Hardsuit LT 1")

################################################################################
# Filtration

T = TraderInstruction
# The opcodes of the cargo entries: items (things and their children) and gases. The item
# entries share a layout, and so do the gas entries.
ITEM_DATA = (
    T.TraderBuyThingData.OP_CODE,
    T.TraderBuyThingChildData.OP_CODE,
    T.TraderSellThingData.OP_CODE,
    T.TraderSellThingChildData.OP_CODE,
)
GAS_DATA = (T.TraderBuyGasData.OP_CODE, T.TraderSellGasData.OP_CODE)
# The dish's stack has 32 addresses.
STACK_END = 32


def cargo_request(at: int, buy: Bool) -> Int:
    """The instruction at address `at` listing the cargo into the addresses after it."""
    first = at + 1
    count = STACK_END - first
    return select(
        buy,
        T.WriteTraderBuyData.pack(write_index=first, write_count=count),
        T.WriteTraderSellData.pack(write_index=first, write_count=count),
    )


@transpile
async def filtration(target: Static[Variable[int]]):
    """Sets `target` (see main) to the signal ID of a contact with matching cargo."""
    # The signal ID made the target last.
    last_locked = 0.0

    while True:
        dish = SatelliteDish(ref_id=DISH_M.ReferenceId)
        filt = ITEM_FILTER.Setting

        # Request: the filter, if any, then the cargo list.
        dish.memory.clear()
        buy = SELL_BUY_SWITCH.Open
        if filt == 0:
            dish.memory[0] = cargo_request(0, buy)
        else:
            # The dish lists only the cargo that matches the filter.
            # the two filters differ only in opcode: the filter goes in bits 8-39 of both
            op = select(
                FILTER_GAS_SWITCH.Open, T.FilterGasContains.OP_CODE, T.FilterPrefabHashEquals.OP_CODE
            )
            dish.memory[0] = (filt << 8) | op
            dish.memory[1] = cargo_request(1, buy)
        await coro_yield()

        signal = DISH_M.SignalID
        LED_SIGID.Setting = signal
        if filt != 0 and target == 0 and signal != last_locked:
            # The dish lists only matching cargo, so any entry after the request is a match.
            if dish.memory[2] != 0:
                target.value = signal
                last_locked = signal

        # The n-th item entry goes to the item memory named with hash ITEM_HASH + n, and so on;
        # the request and the filter before the entries have other opcodes. Entries past the
        # last memory go to hashes that name no device.
        item_name = ITEM_HASH
        gas_name = GAS_HASH
        addr = 0
        while addr < STACK_END - 1:
            addr += 1
            # 8 entries take about 110 lines, and an IC runs at most 128 lines per tick
            if addr % 8 == 0:
                await coro_yield()
            entry = dish.memory[addr]
            op = T.opcode(entry)
            if op == ITEM_DATA[0] or op == ITEM_DATA[1] or op == ITEM_DATA[2] or op == ITEM_DATA[3]:
                LogicMemory(f.transmute(item_name, str)).Setting = T.TraderBuyThingData.prefab_hash(entry)
                item_name += 1
            elif op == GAS_DATA[0] or op == GAS_DATA[1]:
                LogicMemory(f.transmute(gas_name, str)).Setting = T.TraderBuyGasData.gas_types_bitflag(entry)
                gas_name += 1
        # -1 on the memories left over: up to 31 writes, so a part of their own
        await coro_yield()
        if item_name < ITEM_HASH + len(ITEM_MEMORIES):
            while True:
                LogicMemory(f.transmute(item_name, str)).Setting = -1
                item_name += 1
                if item_name >= ITEM_HASH + len(ITEM_MEMORIES):
                    break
        if gas_name < GAS_HASH + len(GAS_MEMORIES):
            while True:
                LogicMemory(f.transmute(gas_name, str)).Setting = -1
                gas_name += 1
                if gas_name >= GAS_HASH + len(GAS_MEMORIES):
                    break


################################################################################
# Dish Control


@static_coroutine
@transpile
async def settle():
    """Tick until the dish is idle; called after aiming it."""
    while True:
        await coro_yield()
        if DISH_M.Idle != 0:
            break


@static_coroutine
@transpile
async def sweep(axis: UserValue[LogicType], step: Int, limit: Int) -> Float:
    """
    Turn the dish `step` degrees along `axis` per tick while it sees the signal and the next
    angle is within 0 to `limit`. Returns the last angle at which it saw the signal.
    """
    seen = DISH_M[axis].get(typ=float)
    while True:
        await coro_yield()
        if DISH_M.SignalID == -1:
            break
        seen = DISH_M[axis].get(typ=float)
        nxt = (seen + step) % 360
        if nxt > limit:
            break
        DISH_M[axis] = nxt
    return seen


@static_coroutine
@transpile
async def center(axis: UserValue[LogicType], limit: Int):
    """
    Find both edges of the signal along `axis` and aim the dish at the middle. `limit` is the
    largest angle of `axis`: horizontal angles wrap around 360; vertical ones stay within 0-90,
    and a step below 0 wraps to above `limit` too.
    """
    low = await sweep(axis, -5, limit)
    DISH_M[axis] = low
    await settle()
    high = await sweep(axis, 5, limit)
    # the upward sweep can wrap past 360 to a smaller angle
    DISH_M[axis] = (low + (high - low) % 360 / 2) % 360
    await settle()


@transpile
async def control(target: Static[Variable[int]]):
    """Repeatedly clears `target` (see main), roams until filtration sets it, then locates and
    interrogates it."""
    DISH_M.On = True
    while True:
        target.value = 0
        # Roam until filtration sets a target.
        DISH_M.Vertical = 60
        DISH_M.BestContactFilter = -1
        DISH_M.Setting = 500
        while True:
            await coro_yield()
            if target != 0:
                break
            DISH_M.Horizontal = (DISH_M.Horizontal + 2) % 360

        DISH_M.BestContactFilter = target
        await settle()
        # If the dish does not see the target here, each sweep ends at once, the dish stays put
        # and the interrogation below ends on its first check.
        await center(DISH_M.Horizontal.logic_type, 360)
        await center(DISH_M.Vertical.logic_type, 90)

        # Interrogate until done or the dish loses the signal.
        while True:
            await coro_yield()
            if DISH_M.SignalID == -1:
                break
            DISH_M.Setting = 50000
            DISH_M.Activate = True
            if DISH_M.InterrogationProgress >= 1:
                TRANSMITTER["SoundAlert"] = 37
                break


@program
def main():
    # The signal ID of the trader to locate and interrogate, 0 for none: set by filtration while
    # it is 0, cleared by Dish Control.
    target = Variable(int)
    filtration_task = create_static_task(filtration(target))
    control_task = create_static_task(control(target))
    # Within a tick filtration runs first, so Dish Control sees a target set in the same tick.
    with loop():
        control_task.step()
        yield_()
        filtration_task.step()


if __name__ == "__main__":
    main.cli()
