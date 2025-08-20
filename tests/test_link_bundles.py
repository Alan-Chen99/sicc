from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@subr()
def func(x: Int) -> None:
    d.Activate = x > 3


@wrap_test
@program()
def test_cond_call():
    """currently function that returns dont work. will hopefully be fixed latter"""
    with if_(d.On):
        func(d.ImportCount)

    with if_(d.Power == 2):
        with else_():
            func(d.ImportCount)


@dataclass
class Data:
    x: Float
    y: Float


@subr()
def child(x: Float):
    with if_(x > 0):
        return_(Data(7, x))
    return Data(x, 9)


@subr()
def parent(x: Float) -> Float:
    with if_(x > 0):
        return_(child(x).x)

    return x + 5


@wrap_test
@program()
def test_nested_call():
    comment("child", child(1).x)
    comment("res1", parent(1))
    comment("res2", parent(2))


dev_name = Variable(str)
ret_to = Variable(Label)


@subr()
def map_devices(func_ptr: ValLabelLike):
    for d in range(5):
        ret_l = mk_label(f"process Autolathe{d} ret")

        dev_name.value = f"Autolathe{d}"
        ret_to.value = ret_l
        jump(func_ptr)
        label(ret_l)


@wrap_test
@program()
def test_function_ptr():

    with if_(False):

        fn1 = label("fn1")
        Autolathe(dev_name).On = True
        jump(ret_to)

        fn2 = label("fn2")
        with if_(Autolathe(dev_name).On):
            Autolathe(dev_name).Activate = True
        jump(ret_to)

    map_devices(fn1)
    map_devices(fn2)
