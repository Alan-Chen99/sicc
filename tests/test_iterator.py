from rich import print as print

from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe("Autolathe1")


@wrap_test
@program
def test_iterator():
    for d in Iterator_.from_iterable([Autolathe(f"Autolathe{i}") for i in range(10)]):
        comment("d:", d.On, d.Error)


@wrap_test
@program
def test_range():
    for i in range_(10):
        for j in range_(i, 10):
            comment("i,j:", i, j)


@wrap_generator
def generator(i: Int):
    for i in range_(i):
        yield i
    for i in range_(10):
        yield i


@wrap_test
@program
def test_generator():
    for j in range_(5, 10):
        for j in generator(j):
            comment("i:", j)
            with if_(j > 8):
                break_()
