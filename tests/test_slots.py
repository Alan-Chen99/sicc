from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program()
def test_devices():
    for d in [VendingMachine(), VendingMachine("somename")]:
        comment(f"{d}:", d.slots[5]["Occupied"].max)
        comment(f"{d}:", d.slots[5]["Quantity"] + 1)
