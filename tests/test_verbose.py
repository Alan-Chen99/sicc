from sicc import *
from sicc.config import verbose
from sicc.devices import *
from test_utils import wrap_test


@subr()
def fn():
    pass


@wrap_test
@program()
def test_verbose():
    verbose.value = 2

    fn()
    fn()

    v = Variable(int)
    with if_(Autolathe().On):
        v.value = 1
        yield_()

        with else_():
            v.value = 2

    comment("res", v)
