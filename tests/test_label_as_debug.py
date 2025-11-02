from sicc import *
from sicc import functions as f
from sicc.devices import *
from test_utils import wrap_test


@subr
def on_err(location: ValLabelLike):
    ConsoleLED5().Setting = location
    exit_program()


def assert_(cond: Bool):
    linenum_label = label_ref("assert_linenum", unique=True)
    with if_(f.not_(cond)):
        on_err(linenum_label)
    label(linenum_label)


d = Autolathe()


@wrap_test
@program
def test_label_as_debug():
    assert_(d.On)
    assert_(d.ImportCount == 0)

    assert_(True)

    # TODO:
    # currently works, but due to a coincidence / order of optimizations
    # it would also be valid to remove the "assert_linenum" label
    # since its unreachable, and therefore writing "undef" to the display
    with if_(d.ExportCount > 2):
        assert_(False)

    assert_(d.ExportCount > 1)


@wrap_test
@program
def test_refer_unreachable_label():
    with if_(False):
        dead = label("dead")
    live = label("live")
    l = cond(ConsoleLED5().On, dead, live)
    comment("l:", l)
