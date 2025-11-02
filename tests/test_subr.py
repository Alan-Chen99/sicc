from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@dataclass
class Data:
    x: Float
    y: Float


@subr
def func_with_ret(x: Int):
    return Data(x + 1, x - 1)


@subr
def func_with_yield(x: Int):
    with if_(x > 0):
        yield Return(Data(1, x))
    yield Return(Data(x, 1))


@subr
def func_with_yield_nested(x: Int) -> FunctionRet[Data]:
    def tmp(x: Int):
        yield Return(Data(x, x))

    with if_(x > 0):
        yield from tmp(x)
    yield Return(Data(x, 1))


@wrap_test
@program
def test_subr():
    for f in [func_with_ret, func_with_yield, func_with_yield_nested]:
        for v in [d.ImportCount, d.ExportCount]:
            ans: Data = f(v)
            comment(f"{f.fn.__qualname__}, {v.logic_type}:", ans)


@subr
def func(x: Int) -> None:
    d.Activate = x > 3


@wrap_test
@program
def test_cond_call():
    """currently function that returns dont work. will hopefully be fixed latter"""
    with if_(d.On):
        func(d.ImportCount)

    with if_(d.Power == 2):
        with else_():
            func(d.ImportCount)


@wrap_test
@program
@subr
def test_subr_in_subr():

    @subr
    def outer(x: Int):

        @subr
        def inner(x: Int):
            return x + 1

        return inner(inner(x))

    with if_(outer(d.ImportCount.transmute(int)) > 0):
        yield Return()

    d.On = False
    yield Return()
