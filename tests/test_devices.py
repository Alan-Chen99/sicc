from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program()
def test_devices():
    all = "all", Autolathe()
    named = "named", Autolathe("Autolathe1")
    default_max = "max", Autolathe("Autolathe2", default_batchmode=BatchMode.MAX)

    for n, d in [all, named, default_max]:
        comment(f"{n}:min:", d.ImportCount.min)
        comment(f"{n}:arith:", d.ExportCount + 1)
        comment(f"{n}:bystr:", d["RequiredPower"])

        d.On = True
        d["Lock"] = True


@wrap_test
@program()
def test_pins():
    comment("db:", db.Temperature)
    db.Setting = 0
    comment("d3:", d3.Temperature)
    d3.Setting = 0

    d_var = State(d0)

    with if_(db.Temperature > 0):
        d_var.value = d1

    d_var.value.Setting = d_var.value.On
