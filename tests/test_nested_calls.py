from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@subr()
def inner(x: Int) -> Int:
    return x + d.ImportCount


@subr()
def outer(x: Int) -> Int:
    orig = x
    for _ in range(5):
        x = inner(x)
    return x + orig * 123


@wrap_test
@program()
def test_nested_call():
    """currently function that returns dont work. will hopefully be fixed latter"""
    x = 0
    for _ in range(5):
        x = outer(x)

    comment("result:", x)
