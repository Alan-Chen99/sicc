from sicc import *
from sicc._iterator import runtime_loop
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe("Autolathe1")


@wrap_test
@program
def test_iterator():
    for d in runtime_loop([Autolathe(f"Autolathe{i}") for i in range(10)]):
        comment("d:", d.On, d.Error)


@wrap_test
@program
def test_range():
    for i in range_(10):
        for j in range_(i, 10):
            comment("i,j:", i, j)


@shared_generator
def generator(i: Int):
    for i in range_(i):
        yield i
    for i in range_(10):
        yield i


@wrap_test
@program
def test_generator():
    for i in range_(5, 10):
        for j in generator(i):
            comment("j:", j)
            with if_(j > 8):
                break_()


@wrap_test
@program
def test_generator2():
    for i in generator(10):
        comment("i1:", i)

    for i in generator(5):
        comment("i2:", i)
