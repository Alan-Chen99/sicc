from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = VolumePump()


@wrap_test
@program
def test_identiy_arith():
    d.Setting = d.Power + 0
    d.Setting = 0 + d.Power

    d.Setting = d.Power - 0
    d.Setting = 0 - d.Power

    d.Setting = d.Power * 1
    d.Setting = 1 * d.Power

    d.Setting = d.Power / 1
    d.Setting = 1 / d.Power
