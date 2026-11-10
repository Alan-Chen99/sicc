"""
Trader shuttle cargo display ("Looking Glasses") and filtration, through the stack of the
Medium Satellite Dish.

Based on the scripts of https://stationeers-wiki.com/Satellite_Tracking, sections
"Shuttle Cargoes Looking Glasses" and "Shuttle Cargoes Auto Filtration/Interrogation and Call
Down". The wiki page describes the devices to build and label, except that the Logic Memories
take the names in ITEM_MEMORIES and GAS_MEMORIES below.

sicc has no builtin for the device stack instructions `clrd`, `putd` and `getd`; this example
emits them with `asm` and `asm_fn`. sicc assumes a raw instruction may read and write any
device, so it neither drops one nor moves it past another device read or write.

Each tick the dish runs the instructions found in its stack and writes their results into it
(see "Stack Usage" on the wiki page Satellite_Tracking). The runtime programs below write two
instructions, one for the strongest contact's metadata and one for its cargo list, then read
the results back:

- metadata goes to the LEDs;
- each cargo entry goes to the next unused Logic Memory of its kind, items and gases
  separately, for a Console with a Hash Display to show. Memories left over are set to -1.

The programs (`sicc examples/dish_trader.py -p <program>`):

- looking_glass: the display.
- filtration: the display, filtered by SLM_ITEM_FILTER. With a filter set, only matching cargo
  is shown, and the signal ID of a trader with matching cargo is written to LED_LOCKED_SIGID,
  for the Dish Control program (dish_control.py) to locate and interrogate it. Nothing is
  written while LED_LOCKED_SIGID is set, nor for the signal ID written last, so a trader is
  not interrogated again once Dish Control clears the LED.
"""

from sicc import *
from sicc.devices import *

################################################################################
# Configuration

DISH_M = SatelliteDish("Medium Satellite Dish 1")

# Logic Memories that receive the cargo, one per Console, labelled with these names. Cargo
# entries beyond these are not shown. The programs write to the n-th memory (from 0) by the name
# hash ITEM_HASH + n or GAS_HASH + n, so they compute each name hash instead of reading a table
# of names. crc_rev(prefix, hash) finds each name: the prefix, then 6 letters and digits that give
# the name that hash. Finding a name takes about 0.2 s, so the names are listed here rather than
# found at each compile; `sicc crc-rev SLMP21_ 1000000020` would name another item memory.
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
LED_CON_TYPE_ID = ConsoleLED5("LED_CON_TYPE_ID")
LED_TIER = ConsoleLED5("LED_TIER")
LED_SHUTTLE_TYPE = ConsoleLED5("LED_SHUTTLE_TYPE")
LED_WATT_TO_RESOLVE = ConsoleLED5("LED_WATT_TO_RESOLVE")
LED_CONTACTED = ConsoleLED5("LED_CONTACTED")
LED_LIFETIME = ConsoleLED5("LED_LIFETIME")

# filtration only:
# The filter: a prefab hash, or gas bitflags when FILTER_GAS_SWITCH is open. 0 for none.
ITEM_FILTER = LogicMemory("SLM_ITEM_FILTER")
FILTER_GAS_SWITCH = LogicSwitch2("FILTER_GAS_SWITCH")
# Output: the signal ID of a trader with matching cargo.
LED_LOCKED_SIGID = ConsoleLED5("LED_LOCKED_SIGID")

################################################################################

# Dish stack opcodes (TraderInstruction.* in IC10), in the low byte of each entry.
WRITE_TRADER_DATA = 1
WRITE_BUY_DATA = 5
WRITE_SELL_DATA = 6
FILTER_PREFAB_EQUALS = 13
FILTER_GAS_CONTAINS = 17
# Cargo entries: items (things and their children) and gases.
ITEM_DATA = (
    7,  # TraderBuyThingData
    8,  # TraderBuyThingChildData
    10,  # TraderSellThingData
    12,  # TraderSellThingChildData
)
GAS_DATA = (
    9,  # TraderBuyGasData
    11,  # TraderSellGasData
)
# The dish's stack has 32 addresses.
STACK_END = 32


# The dish stack instructions, emitted as raw asm. `dish` is the dish's ReferenceId.


def clrd(dish: Int) -> None:
    asm("clrd", [], dish)


def putd(dish: Int, addr: Int, value: Int) -> None:
    asm("putd", [], dish, addr, value)


def getd(dish: Int, addr: Int) -> VarRead[int]:
    entry = Variable(int)
    asm("getd", [entry], dish, addr)
    return entry.value


def cargo_request(at: int, buy: Bool) -> Int:
    """The instruction at address `at` listing the cargo into the addresses after it."""
    first = at + 1
    count = STACK_END - first
    base = (count << 16) | (first << 8)
    return select(buy, base | WRITE_BUY_DATA, base | WRITE_SELL_DATA)


def read_filter(with_filter: bool) -> tuple[Int, Bool]:
    """(filter, is gas filter). Plain python, so looking_glass gets constants and the filter
    code compiles away."""
    if not with_filter:
        return 0, False
    return ITEM_FILTER.Setting, FILTER_GAS_SWITCH.Open


@transpile(inline=True)
def run(with_filter: Static[bool]):
    # The signal ID written to LED_LOCKED_SIGID last.
    last_locked = 0.0
    # The dish stack address read last.
    addr = 0

    # The helpers below read `dish` and `request_at`, set by the main loop.

    @transpile(inline=True)
    def next_cargo(ops: Static[tuple[int, ...]]) -> Int:
        """The value of the next entry after `addr` with an opcode in `ops`; -1 past the last."""
        nonlocal addr
        while addr < STACK_END - 1:
            addr += 1
            entry = getd(dish, addr)
            op = entry & 0xFF
            for wanted in ops:
                if op == wanted:
                    return entry >> 16
        return -1

    @transpile(inline=True)
    def show_cargo(first_hash: Static[int], count: Static[int], ops: Static[tuple[int, ...]]):
        """Show the cargo entries with an opcode in `ops` on the `count` Logic Memories whose
        name hashes start at `first_hash`, in order; -1 on those left over."""
        nonlocal addr
        addr = request_at
        for name in range_(first_hash, first_hash + count):
            LogicMemory(name.transmute(str)).Setting = next_cargo(ops)

    while True:
        dish = DISH_M.ReferenceId
        if dish == nan:
            # no dish of that name
            yield_()
            continue
        filt, filt_gas = read_filter(with_filter)

        # Request: metadata into addresses 1-3; then the filter, if any; then the cargo list.
        clrd(dish)
        putd(dish, 0, (1 << 8) | WRITE_TRADER_DATA)
        buy = SELL_BUY_SWITCH.Open
        if filt == 0:
            request_at = 4
            putd(dish, 4, cargo_request(4, buy))
        else:
            # The dish lists only the cargo that matches the filter.
            request_at = 5
            filter_op = select(filt_gas, FILTER_GAS_CONTAINS, FILTER_PREFAB_EQUALS)
            putd(dish, 4, (filt << 8) | filter_op)
            putd(dish, 5, cargo_request(5, buy))
        yield_()

        signal = DISH_M.SignalID
        LED_SIGID.Setting = signal
        LED_CON_TYPE_ID.Setting = getd(dish, 1) >> 8
        meta = getd(dish, 2)
        LED_CONTACTED.Setting = (meta >> 24) & 0xFF
        LED_TIER.Setting = (meta >> 16) & 0xFF
        LED_SHUTTLE_TYPE.Setting = (meta >> 8) & 0xFF
        signal_data = getd(dish, 3)
        LED_WATT_TO_RESOLVE.Setting = (signal_data >> 8) & 0xFFFF
        LED_LIFETIME.Setting = (signal_data >> 24) & 0xFFFF

        show_cargo(ITEM_HASH, len(ITEM_MEMORIES), ITEM_DATA)
        show_cargo(GAS_HASH, len(GAS_MEMORIES), GAS_DATA)

        if filt != 0 and LED_LOCKED_SIGID.Setting == 0 and signal != last_locked:
            # The dish lists only matching cargo, so any entry after the request is a match.
            if getd(dish, request_at + 1) != 0:
                LED_LOCKED_SIGID.Setting = signal
                last_locked = signal


@program
@transpile
def looking_glass():
    run(False)


@program
@transpile
def filtration():
    run(True)


if __name__ == "__main__":
    looking_glass.cli()
