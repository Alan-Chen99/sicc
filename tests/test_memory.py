from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program
def test_memory():
    # on a pin: get, put, clr
    sorter = LogicSorter(pin=0)
    sorter.memory.clear()
    sorter.memory[0] = SorterInstruction.FilterPrefabHashEquals.pack("ItemIronIngot")
    sorter.memory[1] = SorterInstruction.LimitNextExecutionByCount.pack(count=3)
    comment("first op:", SorterInstruction.opcode(sorter.memory[0]))

    # by name: getd, putd, clrd through the device's ReferenceId, read each time
    named = LogicSorter("sorter")
    named.memory.clear()
    named.memory[2] = sorter.memory[1]

    # by ReferenceId, read once
    dish = SatelliteDish(ref_id=Variable(SatelliteDish("dish").ReferenceId))
    dish.memory[0] = TraderInstruction.WriteTraderData.pack(write_index=1)
    comment("contact:", TraderInstruction.StrongestContactIdHash.trader_id_hash(dish.memory[2]))

    # another chip's stack holds whatever its program put there
    other = CircuitHousing("other")
    other.memory[10] = 1.5
    comment("other:", other.memory[11].transmute(float) + 1)
    # so does a pin's memory, untyped
    d3.memory[0] = 2
    comment("d3:", d3.memory[1].transmute(float))


@wrap_test
@program
def test_instructions():
    dish = SatelliteDish(pin=1)
    # compile-time fields fold into a constant; runtime ones are masked and shifted in
    count = Variable(dish.ContactSlotIndex)
    dish.memory[4] = TraderInstruction.WriteTraderBuyData.pack(write_index=5, write_count=count)
    dish.memory[5] = TraderInstruction.FilterPrefabHashEquals.pack(Variable(str, "ItemIronIngot"))

    # decoding a cargo entry: the hash comes back signed, as the game reports it unsigned
    entry = Variable(dish.memory[6])
    buy = TraderInstruction.opcode(entry) == TraderInstruction.TraderBuyThingData.OP_CODE
    quantity = TraderInstruction.TraderBuyThingData.quantity(entry)
    iron = TraderInstruction.TraderBuyThingData.prefab_hash(entry) == "ItemIronIngot"
    comment("buy:", buy, "quantity:", quantity, "iron:", iron)

    # a bool field
    printer = Autolathe(pin=2)
    printer.memory[0] = PrinterInstruction.DeviceSetLock.pack(lock_state=printer.Lock)
