from sicc import *
from sicc.devices import *
from test_utils import wrap_test

d = Autolathe()


@wrap_test
@program
def test_nan():
    with if_(d.On == nan):
        comment("is nan1:", d.On)
    with if_(d.On != nan):
        comment("not nan2:", d.On)
    with if_(d.On.is_nan()):
        comment("is nan3:", d.On)
