from rich import print as print

from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@wrap_test
@program()
def test_stack_devices():
    for pin in [db, d1]:
        ptr = stack_var(float, pin)
        comment(f"{pin}: ptr:", ptr)
        comment(f"{pin}: val:", ptr.read())
        ptr.write(5)
        comment(f"{pin}: val:", ptr.read())


@dataclass
class Data:
    x: Int
    y: tuple[Int, Float]


@wrap_test
@program()
def test_class():
    ptr = stack_var(Data(1, (2, 3.0)))
    comment("ptr:", ptr)
    comment("val:", ptr.read())

    y_ptr = ptr.project(lambda d: d.y)
    comment("x:", y_ptr, y_ptr.read())


@wrap_test
@program()
def test_arr_idx():
    ptr = stack_var([Data(i, (undef(int), undef(float))) for i in range(10)])

    for i in range_(10):
        d = ptr[i]
        v = d.read()
        comment("value:", d, v)
        d.write(Data(v.y[0], (v.x, v.x)))


@wrap_test
@program()
def test_arr_iter():
    ptr = stack_var([Data(i, (undef(int), undef(float))) for i in range(10)])

    for d in ptr:
        v = d.read()
        comment("value:", d, v)
        d.write(Data(v.y[0], (v.x, v.x)))


@wrap_test
@program()
def test_ptr_project():
    ptr = stack_var([Data(i, (undef(int), undef(float))) for i in range(10)])

    for d in ptr:
        dy = d.project(lambda d: d.y)
        v = dy.read()
        comment("value:", v)
        dy.write((v[0], v[0]))
