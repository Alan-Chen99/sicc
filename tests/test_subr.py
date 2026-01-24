from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@dataclass
class Data:
    x: Float
    y: Float


@subr
def func_with_ret_py(x: Int) -> FunctionRet[Data]:
    return Data(x + 1, x - 1)


@subr
def func_with_return_(x: Int) -> FunctionRet[Data]:
    with if_(x > 0):
        return_(Data(1, x))
    return_(Data(x, 1))


@wrap_test
@program
def test_subr():
    for f in [func_with_ret_py, func_with_return_]:
        for v in [d.ImportCount, d.ExportCount]:
            ans: Data = f(v)
            comment(f"{f.__qualname__}, {v.logic_type}:", ans)


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
def test_subr_in_subr() -> FunctionRet[None]:

    @subr
    def outer(x: Int):

        @subr
        def inner(x: Int):
            return x + 1

        return inner(inner(x))

    with if_(outer(d.ImportCount.transmute(int)) > 0):
        return_()

    d.On = False
    return_()
