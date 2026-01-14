from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@subr
def fn1() -> FunctionRet[tuple[Float, Float, Bool]]:
    comment("fn1")
    with if_(d.On):
        yield Return((5.0, True, True))

    yield Return((True, 5.0, d.On))


@subr
def fn2() -> FunctionRet[tuple[Float, Float, Bool]]:
    comment("fn2")
    with if_(d.On):
        return_((6.0, True, True))

    return_((True, 6.0, d.On))


@wrap_test
@program()
def test_return_type_promotion():
    comment("result1:", *fn1())
    comment("result2:", *fn2())
