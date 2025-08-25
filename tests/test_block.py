from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@wrap_test
@program()
def test_block_type_promote():

    with block(float) as b:
        with if_(d.On):
            b.break_(False)
        b.break_(1.1)
    comment("ans1:", b.value)

    with block(float) as b:
        with if_(d.On):
            b.break_(1.1)
        b.break_(False)
    comment("ans2:", b.value)

    with block() as b:
        with loop():
            yield_()
            with if_(d.On):
                b.break_()

    comment("on!")
