from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@wrap_test
@program()
def test_asm_block():
    x = Variable(d.On)
    y = Variable(x)

    asm_block(
        ("add", x, y, 1),
        ("jltr", x, 0, -1),
    )

    comment("result", x)
