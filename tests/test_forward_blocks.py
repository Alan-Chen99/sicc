from sicc import *
from sicc import functions as f
from test_utils import wrap_test


@wrap_test
@program()
def test_forward():
    x = Variable(int)

    with if_(f.black_box(True)):
        x.value = 1

        with else_():
            x.value = 2

    comment("x:", x)


@wrap_test
@program(loop=True)
def test_forward_fn():
    @subr()
    def fn(x: Int) -> Int:
        return x + 1

    comment("call1", fn(1))

    v = Variable(1)
    with if_(f.black_box(True)):
        # we dont move the +123 into the subr atm
        # we could save a line here but this opt
        # is not always helpful
        v.value = fn(2) + 123

    comment("res:", v)
