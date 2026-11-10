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


@wrap_test
@program
def test_mod():
    # float operands: mod on floats, as in IC10
    d.Setting = d.Power % 360
    d.Setting = (d.Power + 0.5) % 2.5
    # folded at compile time: expected Setting 1.5 and 1
    d.Setting = Variable(7.5) % 2
    d.Setting = Variable(7) % 3
    # int operands keep an int result
    i = Variable(int, 7)
    i.value = i % 4
    d.Setting = i


@wrap_test
@program
def test_neg():
    d.Setting = -d.Power
    # folded at compile time: expected Setting -2.5
    d.Setting = -Variable(2.5)
    # an int stays an int
    i = Variable(d.RequiredPower)
    i.value = -i
    d.Setting = i
