from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program
def test_fence_busy_wait():
    # without the fence, the condition's read merges into `start` and the loop never exits;
    # the fence leaves no jump in the loop body
    start = Variable(float, d0.Setting)
    with while_(lambda: d0.Setting == start):
        fence()
    d1.Setting = 1


@wrap_test
@program
def test_fence_reread():
    a = Variable(float, d0.Setting)
    fence()
    b = Variable(float, d0.Setting)
    d1.Setting = a + b


@wrap_test
@program
def test_fence_guarded_read():
    # without the fence, the ld runs before the branch, also when there is no dish
    dish_id = Variable(SatelliteDish("dish").ReferenceId)
    v = Variable(float)
    with if_(dish_id == nan):
        v.value = 0.0
        with else_():
            fence()
            v.value = SatelliteDish(ref_id=dish_id).Horizontal
    d1.Setting = v
