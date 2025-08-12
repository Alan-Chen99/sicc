from sicc import *
from sicc import functions as f
from test_utils import wrap_test


@wrap_test
@program()
def test_if_else():
    with if_(f.black_box(True)):
        comment("true_branch")

        with else_():
            comment("false_branch")


@wrap_test
@program()
def test_if_only():
    with if_(f.black_box(True)):
        comment("true_branch")


@wrap_test
@program()
def test_else_only():
    with if_(f.black_box(True)):
        with else_():
            comment("false_branch")
