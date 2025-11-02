from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@subr
def fn() -> FunctionRet[tuple[Float, Float, Bool]]:
    with if_(d.On):
        yield Return((5.0, True, True))

    yield Return((True, 5.0, d.On))


@wrap_test
@program()
def test_return_type_promotion():
    comment("result:", *fn())
