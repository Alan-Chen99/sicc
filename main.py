# pyright: reportUnusedImport=false

from rich import print

from sicc import *
from sicc import functions as f
from sicc.devices import Autolathe


@subr()
def error():
    exit_program()


@subr()
def check_device(d: Autolathe):
    with if_(~d["On", BatchMode.MAX]):
        error()
    with if_(~d.Open):
        error()
    with if_(~d["Lock"]):
        error()
    with if_(~d.On.sum):
        error()


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
        _ = x + 1
        return_(child(x).x)

    return x + 5


@program(
    loop=True,
)
def main():

    a = Autolathe()["X"].avg
    y = Autolathe()["Y"].avg

    x = Variable(int)

    asm_block(
        ("raw1", x, a.value, y),
        # ("raw2", a, y, Variable(bool), x),
    )

    comment("result", x)

    return

    # d1, d2, d3, d4 = [Autolathe(f"MyAutolathe{i}") for i in range(4)]

    # print("d1", d1)

    # check_device(d1)
    # check_device(d2)
    # check_device(d3)
    # check_device(d4)

    # with while_(lambda: parent(1) > 0):
    #     yield_()
    #     with if_(~d1.On):
    #         break_()


if __name__ == "__main__":
    main.cli()
