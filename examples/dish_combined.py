"""
Trader filtration and Dish Control on one IC: the program `filtration` of dish_trader.py and
dish_control.py in one program, on the Medium Satellite Dish DISH_M.

Dish Control is the main program. Filtration runs as a static coroutine: wait() steps it after
each yield_(), and it hands control back once it has written its next request. The two talk
through the LED LED_LOCKED_SIGID, as the separate programs do. Reading the cargo back scans the
dish's stack twice, a few hundred lines, while an IC runs at most 128 lines per tick; so each
wait() takes a few ticks, and Dish Control steps that much slower than dish_control.py.

Filtration (see dish_trader.py, which describes the devices to build and label): each round
writes a request for the strongest contact's cargo into the dish's stack, filtered by
SLM_ITEM_FILTER if set, and reads the cargo back the next tick onto the Logic Memories named in
ITEM_MEMORIES and GAS_MEMORIES. With a filter set, the signal ID of a trader with matching
cargo is written to LED_LOCKED_SIGID; nothing is written while the LED is set, nor for the
signal ID written last.

Dish Control (see dish_control.py): while LED_LOCKED_SIGID reads 0, the dish roams at vertical
60, turning 2 degrees per step. When the LED shows a signal ID, the dish finds both horizontal
edges of the signal, then both vertical edges, aims at the middle and interrogates the signal.
On completion the transmitter sounds an alert and the LED is cleared; the LED is cleared too if
the dish loses the signal during the interrogation. Changing the LED abandons the search. Each
step waits for the dish to be idle.

Dropped to fit in 128 lines:
- filtration: the contact's metadata on the LEDs LED_CON_TYPE_ID, LED_TIER, LED_SHUTTLE_TYPE,
  LED_WATT_TO_RESOLVE, LED_CONTACTED and LED_LIFETIME (LED_SIGID still shows the contact)
- filtration: waiting for a dish named as DISH_M to exist; without one, the IC stops with an
  error
- Dish Control: the optional scan and comm dishes (DISH_S, DISH_L), as DISH_M does both; the
  2 s sleep before the alert, which would stall filtration too

Every device is addressed by name over the data network, so the IC housing needs no pins set.
"""

from typing import Any
from typing import Callable

from sicc import *
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
# The signal ID of a trader to locate and interrogate, 0 for none: written by filtration (or by
# hand), cleared by Dish Control when an interrogation ends.
LED_LOCKED_SIGID = ConsoleLED5("LED_LOCKED_SIGID")
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
async def filtration():
    # The signal ID written to LED_LOCKED_SIGID last.
    last_locked = 0.0
    # The dish stack address read last.
    addr = 0

    # The helpers below read `dish` and `request_at`, set by the main loop.

    @transpile(inline=True)
    def next_cargo(ops: Static[tuple[int, ...]], field: Static[Callable[[Int], VarRead[Any]]]) -> Int:
        """`field` of the next entry after `addr` with an opcode in `ops`; -1 past the last."""
        nonlocal addr
        while addr < STACK_END - 1:
            addr += 1
            entry = dish.memory[addr]
            op = T.opcode(entry)
            for wanted in ops:
                if op == wanted:
                    # a number, as is the -1
                    return field(entry).transmute(int)
        return -1

    @transpile(inline=True)
    def show_cargo(
        first_hash: Static[int],
        count: Static[int],
        ops: Static[tuple[int, ...]],
        field: Static[Callable[[Int], VarRead[Any]]],
    ):
        """Show `field` of the cargo entries with an opcode in `ops` on the `count` Logic Memories
        whose name hashes start at `first_hash`, in order; -1 on those left over."""
        nonlocal addr
        addr = request_at
        for name in range_(first_hash, first_hash + count):
            LogicMemory(name.transmute(str)).Setting = next_cargo(ops, field)

    while True:
        dish = SatelliteDish(ref_id=DISH_M.ReferenceId)
        filt = ITEM_FILTER.Setting

        # Request: the filter, if any, then the cargo list.
        dish.memory.clear()
        buy = SELL_BUY_SWITCH.Open
        if filt == 0:
            request_at = 0
            dish.memory[0] = cargo_request(0, buy)
        else:
            # The dish lists only the cargo that matches the filter.
            request_at = 1
            # the two filters differ only in opcode: the filter goes in bits 8-39 of both
            op = select(
                FILTER_GAS_SWITCH.Open, T.FilterGasContains.OP_CODE, T.FilterPrefabHashEquals.OP_CODE
            )
            dish.memory[0] = (filt << 8) | op
            dish.memory[1] = cargo_request(1, buy)
        await coro_yield()

        signal = DISH_M.SignalID
        LED_SIGID.Setting = signal
        show_cargo(ITEM_HASH, len(ITEM_MEMORIES), ITEM_DATA, T.TraderBuyThingData.prefab_hash)
        show_cargo(GAS_HASH, len(GAS_MEMORIES), GAS_DATA, T.TraderBuyGasData.gas_types_bitflag)

        if filt != 0 and LED_LOCKED_SIGID.Setting == 0 and signal != last_locked:
            # The dish lists only matching cargo, so any entry after the request is a match.
            if dish.memory[request_at + 1] != 0:
                LED_LOCKED_SIGID.Setting = signal
                last_locked = signal


################################################################################
# Dish Control

# The signal ID being searched for; set by control, read by wait().
target = Variable(int)
# Start of control's main loop; wait() jumps here to abandon a search.
RESTART = label_ref("restart")
# The filtration task; created by main.
filtration_task: StaticTask


@transpile
def wait():
    """
    Tick, stepping filtration, until the dish is idle. If LED_LOCKED_SIGID no longer shows
    `target`, the search is over: jump to RESTART instead of returning.
    """
    while True:
        yield_()
        filtration_task.step()
        if LED_LOCKED_SIGID.Setting != target:
            jump(RESTART)
        if DISH_M.Idle != 0:
            break


@transpile(inline=True)
def sweep(axis: UserValue[LogicType], start: Float, step: Static[int], limit: Int) -> Float:
    """
    Aim the dish's `axis` at `start`, then `step` degrees further each wait() while it sees the
    signal. Returns the first angle, mod 360, at which it does not or that exceeds `limit`.
    """
    pos = start
    while True:
        DISH_M[axis] = pos
        wait()
        if DISH_M.SignalID == -1:
            break
        pos = (pos + step) % 360
        if pos > limit:
            break
    return pos


@transpile
def center(axis: UserValue[LogicType], limit: Int):
    """
    Find both edges of the signal along `axis` and aim the dish at the middle. `limit` is the
    largest angle of `axis`: horizontal angles wrap around 360, vertical ones stay within 0-90,
    where a step below 0 wraps to above `limit` too.
    """
    # Each sweep returns one step past an edge.
    low = (sweep(axis, DISH_M[axis].get(typ=float), -5, limit) + 5) % 360
    high = sweep(axis, low, 5, limit) - 5
    # the upward sweep can wrap past 360 to a smaller angle
    DISH_M[axis] = (low + (high - low) % 360 / 2) % 360


@transpile(inline=True)
def control():
    DISH_M.On = True
    while True:
        label(RESTART)
        target.value = LED_LOCKED_SIGID.Setting
        if target == 0:
            DISH_M.Vertical = 60
            DISH_M.BestContactFilter = -1
            DISH_M.Setting = 500
            while True:
                wait()
                DISH_M.Horizontal = (DISH_M.Horizontal + 2) % 360

        DISH_M.BestContactFilter = target
        while True:
            wait()
            if DISH_M.SignalID != -1:
                break
        center(DISH_M.Horizontal.logic_type, 360)
        center(DISH_M.Vertical.logic_type, 90)

        # Interrogate until done or the dish loses the signal.
        while True:
            wait()
            if DISH_M.SignalID == -1:
                break
            DISH_M.Setting = 50000
            DISH_M.Activate = True
            if DISH_M.InterrogationProgress >= 1:
                TRANSMITTER["SoundAlert"] = 37
                break
        LED_LOCKED_SIGID.Setting = 0


@program
def main():
    global filtration_task
    filtration_task = create_static_task(filtration())
    control()


if __name__ == "__main__":
    main.cli()
