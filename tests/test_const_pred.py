from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program
@transpile
def test_const_pred():
    # transpiled locals are runtime variables; they become constants during optimization
    i = 5
    x = 0.5
    s = "abc"
    n = nan

    # expected to remain, without branches: Setting 1, 4, 5, 6, 7
    if i == 5:
        d0.Setting = 1
    if i != 5:
        d0.Setting = 2
    if x < 0.5:
        d0.Setting = 3
    if i <= 5.0:
        d0.Setting = 4
    if s == "abc":
        d0.Setting = 5
    if s != "abd":  # pyright: ignore[reportUnnecessaryComparison]
        d0.Setting = 6
    if n == nan:
        d0.Setting = 7
    if n != nan:
        d0.Setting = 8
    if i == nan:
        d0.Setting = 9


@wrap_test
@program
def test_const_pred_label():
    a = label_ref("a")
    b = label_ref("b")
    c = label_ref("c")
    # a and b mark the same position and are merged into one label: expected to fold
    with if_(Variable(a) != b):
        d0.Setting = 1
    # distinct labels are only resolved to lines when emitting: expected to remain
    with if_(Variable(a) == c):
        d0.Setting = 2
    label(a)
    label(b)
    d0.Setting = 3
    label(c)
    d0.Setting = 4
