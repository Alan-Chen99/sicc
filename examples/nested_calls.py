from sicc import *
from sicc.devices import *

d = Autolathe()


@transpile
def inner(x: Int) -> Int:
    return x + d.ImportCount


@transpile
def outer(x: Int) -> Int:
    return inner(inner(inner(x)))


@program
@transpile
def main():
    x = inner(0)
    for _ in range(5):
        x = outer(x)
    comment("result:", x)


if __name__ == "__main__":
    main.cli()
