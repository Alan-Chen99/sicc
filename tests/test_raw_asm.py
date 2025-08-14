from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@wrap_test
@program()
def test_asm_block():
    x = Variable(float, d.ExportCount)
    y = Variable(x)
    z = Variable(int)

    with if_(False):
        out = label("out")
        x.value += 1

    asm_block(
        ("move", z, 10),
        ("add", x, y, z),
        ("jlt", x, 0, out),
    )

    comment("result", x, y)
