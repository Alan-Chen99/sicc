# pyright: reportUnusedImport=false

from sicc import *
from sicc import functions as f
from sicc.devices import Autolathe


@subr()
def error():
    exit_program()


@subr()
def check_device(d: Autolathe):
    with if_(~d["On", BatchMode.AVG]):
        error()
    with if_(~d.Open.avg):
        error()
    with if_(~d["Lock"].avg):
        error()
    with if_(~d.On.avg):
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
        return_(child(x).x)

    return x + 5


@program()
def main():
    label("start")

    # d1, d2, d3, d4 = [Device("Autolathe", f"MyAutolathe{i}") for i in range(4)]
    d1, d2, d3, d4 = [Autolathe(f"MyAutolathe{i}") for i in range(4)]

    check_device(d1)
    check_device(d2)
    check_device(d3)
    check_device(d4)

    f.black_box(Autolathe().Lock.avg)

    # x = black_box(child(1).y)
    # # x = black_box(parent(1))

    with while_(lambda: parent(1) > 0):
        with if_(~d1.On.avg):
            break_()

    # parent(1)

    # # with trace_bundle():

    # with if_(black_box(x > 2)):
    #     black_box(parent(x))

    # black_box(55)

    jump("start")


if __name__ == "__main__":
    main.cli()
