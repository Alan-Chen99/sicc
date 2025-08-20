# sicc does not have tail call opt
# previously remove_trivial_blocks did tail call opt unintentionally (and incorrectly)
# so we put this test here

from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@subr()
def inner():
    return


@subr()
def outer():
    inner()
    inner()


@wrap_test
@program()
def test_tail_call():
    outer()
    outer()
