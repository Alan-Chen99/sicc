from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = GlassDoor("MyDoor")


@wrap_test
@program
def test_while():
    with while_(lambda: d.Power):
        with if_(d.Open):
            continue_()
        yield_()


@wrap_test
@program
def test_loop():
    with loop():
        yield_()
        with if_(~d.Open):
            break_()


@wrap_test
@program
def test_range():
    for i in range_(12):
        with if_(i == 5):
            yield Return()
            continue_()
        yield_()

    for i in range_(34, 56):
        with if_(i == 50):
            break_()
        yield_()

    yield Return()
