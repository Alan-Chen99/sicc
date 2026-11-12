from sicc import *
from sicc.devices import *
from test_utils import wrap_test

light1 = Diode("Diode1")
light2 = Diode("Diode2")


async def task_1():
    with loop():
        # await shared()
        light1.Color = Color.Red
        await coro_yield()
        light1.Color = Color.Green
        await coro_yield()
        # await shared()


async def task_2():
    with loop():
        light2.Color = Color.Blue
        await coro_yield()
        light2.Color = Color.Orange
        await coro_yield()
        light2.Color = Color.Gray
        await coro_yield()


@wrap_test
@program
def test_simple():
    task1 = create_static_task(task_1())

    with loop():
        light2.Color = Color.Blue
        yield_()
        task1.step()
        light2.Color = Color.Orange
        yield_()


@wrap_test
@program
def test_concur():
    task1 = create_static_task(task_1())
    task2 = create_static_task(task_2())

    with loop():
        yield_()
        task1.step()
        task2.step()


@static_coroutine
async def color_next_tick():
    await coro_yield()
    return light1.Color


async def task_copy():
    with loop():
        light2.Color = await color_next_tick()


@wrap_test
@program
def test_return_value():
    task = create_static_task(task_copy())

    with loop():
        yield_()
        task.step()


display = ConsoleLED5("Display")


@transpile
async def ticks_until_on(step: Static[int]) -> Int:
    # inlined into the coroutine awaiting it; `return` leaves this helper, not that coroutine
    n = 0
    while True:
        await coro_yield()
        if light1.On:
            return n
        n += step


@transpile
async def task_ticks():
    while True:
        display.Setting = await ticks_until_on(10)


@wrap_test
@program
def test_transpiled_helper():
    task = create_static_task(task_ticks())

    with loop():
        yield_()
        task.step()


async def task_two_values():
    with loop():
        a = Variable(float, display.Setting)
        await coro_yield()
        light1.On = a > 1
        b = Variable(float, display.Setting)
        await coro_yield()
        light2.On = b > 2


@wrap_test
@program
def test_live_across_different_yields():
    # `a` is live across the first yield and `b` across the second: they can share a register
    task = create_static_task(task_two_values())

    with loop():
        yield_()
        task.step()
