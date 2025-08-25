from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = GlassDoor("MyDoor")


@wrap_test
@program()
def test_loops():
    with while_(lambda: d["X"]):
        yield_()

    with loop():
        yield_()
        with if_(~d["Y"]):
            break_()


@wrap_test
@program()
def test_select():
    comment("select", select(d.Idle, d.Power, 5))


@wrap_test
@program()
def test_cond():
    comment("cond_consts", cond(d["C1"], 1, 2))
    comment("cond_const_fn", cond(d["C2"], lambda: 1, lambda: False))
    comment("cond_device_var", cond(d["C3"], lambda: d["X"], lambda: d["Y"]))
