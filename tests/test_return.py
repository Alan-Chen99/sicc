from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@subr
def fn() -> tuple[Float, Float, Bool]:
    with if_(d.On):
        return_((5.0, True, True))

    return True, 5.0, d.On


@wrap_test
@program()
def test_return_type_promotion():
    comment("result:", *fn())
