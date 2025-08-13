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
