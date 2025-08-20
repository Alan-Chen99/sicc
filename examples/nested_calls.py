from sicc import *
from sicc.devices import *

d = Autolathe()


@subr
def inner(x: Int) -> Int:
    return x + d.ImportCount


@subr
def outer(x: Int) -> Int:
    return inner(inner(inner(x)))


@program
def main():
    x = inner(0)
    for _ in range(5):
        x = outer(x)
    comment("result:", x)


if __name__ == "__main__":
    main.cli()
