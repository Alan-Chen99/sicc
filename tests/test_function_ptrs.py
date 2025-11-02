from sicc import *
from sicc.devices import *
from test_utils import wrap_test

dev_name = Variable(str)
ret_to = Variable(Label)


@subr()
def map_devices(func_ptr: ValLabelLike):
    for d in range(5):
        ret_l = label_ref(f"process Autolathe{d} ret")

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
