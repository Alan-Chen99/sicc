from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program()
def test_devices():
    for d in [VendingMachine(), VendingMachine("somename")]:
        comment(f"{d}:", d.slots[5]["Occupied"].max)
        comment(f"{d}:", d.slots[5]["Quantity"] + 1)


@wrap_test
@program()
def test_typed_slots():
    # typed reads: every vending machine (batch), a named one, a pin, a ReferenceId
    vm = VendingMachine()
    comment("occupied:", vm.slots[5].Occupied.max, "quantity:", vm.slots[5].Quantity + 1)
    comment("iron:", VendingMachine("vm").slots[2].OccupantHash == "ItemIronIngot")
    tray = HydroponicsTrayData(pin=1)
    # Mature is -1 without a plant
    comment("no plant:", tray.slots[0].Mature == -1, "stage:", tray.slots[0].Growth)
    # a runtime index is not checked against the device's slots
    idx = Variable(vm.DispenseSlot)
    comment("dispensing:", vm.slots[idx].PrefabHash)

    # writes: ss on a pin; sbs on every device of the type; ss by ReferenceId on a named one,
    # as IC10 has no named batch slot write
    GasTankStorage(pin=2).slots[0].Open = True
    GasTankStorage().slots[0].Open = False
    GasTankStorage("tank").slots[0].Open = True
    tank = GasTankStorage(ref_id=Variable(GasTankStorage("tank2").ReferenceId))
    tank.slots[0].Open = ~tank.slots[0].Open
    comment("tank:", tank.slots[0].Pressure)
