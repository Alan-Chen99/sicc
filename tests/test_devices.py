from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program
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
@program
def test_pins():
    comment("db:", db.Temperature)
    db.Setting = 0
    comment("d3:", d3.Temperature)
    d3.Setting = 0

    d_var = State(d0)

    with if_(db.Temperature > 0):
        d_var.value = d1

    d_var.value.Setting = d_var.value.On


filt = Filtration("filt1")


def port_fn(x: GasPort):
    return x.RatioLiquidCarbonDioxide + x.RatioCarbonDioxide


@wrap_test
@program
def test_gas_ports():
    comment("input:", port_fn(filt.input))
    comment("output:", port_fn(filt.output))
    comment("output2:", port_fn(filt.output2))


@wrap_test
@program
def test_prefab_names():
    # prefab name without the "Structure" prefix
    comment("stopwatch:", StopWatch().Time)
    # prefab name that is not "Structure" + a Python identifier
    TransformerMediumReversed("T1").Setting = 100
    comment("hydrogen:", filt.output.RatioHydrogen)
