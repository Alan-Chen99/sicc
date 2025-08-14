from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@wrap_test
@program()
def test_possible_uninit():
    x = Variable(int)

    with loop():
        val = d.Lock.avg
        with if_(d.On):
            x.value = val

        comment("x:", x)


@wrap_test
@program()
def test_swap():
    x = Variable(int, d["X"])
    y = Variable(int, d["Y"])

    with if_(d.Lock):
        x.value, y.value = y.value, x.value

    comment("vals1:", x, y)

    with loop():
        x.value, y.value = y.value, x.value

        comment("vals2:", x, y)
