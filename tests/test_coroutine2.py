from sicc import *
from sicc import functions as f
from sicc._coroutine2 import CoroBuffer
from sicc._coroutine2 import Task
from sicc._coroutine2 import coroutine
from sicc._coroutine2 import create_task
from sicc._coroutine2 import suspend
from sicc.devices import *
from test_utils import wrap_test

light1 = Diode("Diode1")
light2 = Diode("Diode2")


# @static_coroutine
# async def shared():
#     comment("shared...")
#     await suspend()
#     comment("shared")


@coroutine
async def task_1():
    with loop():
        # await shared()
        light1.Color = Color.Red
        await suspend()
        light1.Color = Color.Green
        await suspend()
        # await shared()


@coroutine
async def task_2():
    with loop():
        # await shared()
        light2.Color = Color.Blue
        await suspend()
        light2.Color = Color.Purple
        await suspend()
        # await shared()


@wrap_test
@program
def test_two_task():
    task1 = create_task(task_1(), stack_var(CoroBuffer.create(10)))
    task2 = create_task(task_2(), stack_var(CoroBuffer.create(10)))

    with loop():
        task1.step()
        task2.step()


@coroutine
async def operate(l: Diode):
    with loop():
        l.Color = Color.Red
        await suspend()
        l.Color = Color.Green
        await suspend()


@wrap_test
@program
def test_task_arr():
    arr = stack_var([CoroBuffer.create(5) for _ in range(5)])

    for i in range_(5):
        create_task(operate(Diode(f.transmute(i))), arr[i])

    with loop():
        for b in arr.iter_mut():
            Task(b).step()
