from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program()
def test_always_uninit():
    x = Variable(int)

    comment("x:", x)
