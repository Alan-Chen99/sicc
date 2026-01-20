from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Coroutine
from typing import Generator
from typing import Generic
from typing import TypeVar

from ._api import ValLabelLike
from ._api import Variable
from ._api import jump
from ._api import unreachable_checked
from ._control_flow import if_
from ._core import Label
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._state import State
from ._subr import subr
from ._tracing import exit_program
from ._tracing import label
from ._tracing import mk_internal_label
from ._tree_utils import copy_tree
from ._utils import CallDescMixin

register_exclusion(__file__)


class CoroYield:
    def __await__(self):
        yield self


# seperate function so that you get a warning if you dont await
async def coro_yield() -> None:
    await CoroYield()


F_co = TypeVar("F_co", covariant=True)


@dataclass
class StaticCoroFn(Generic[F_co], CallDescMixin):
    _handler: F_co

    def __call__[**P, R](
        self: StaticCoroFn[Callable[P, R]], *args: P.args, **kwargs: P.kwargs
    ) -> StaticCoro[R]:
        args, kwargs = copy_tree((args, kwargs))
        return StaticCoro(self._handler, args, kwargs)


@dataclass
class StaticCoro[R](Awaitable[R]):
    handler: Callable[..., R]
    args: Any
    kwargs: Any

    def __await__(self) -> Generator[Any, Any, R]:
        with track_caller():
            ans = yield self
        return ans


global_suspend_to: Variable[Label] = Variable(Label)
global_resume_to: Variable[Label] = Variable(Label)


@dataclass
class StaticTask:
    state: State[ValLabelLike]

    def step(self) -> None:
        suspend_to = mk_internal_label("suspend_to")

        global_suspend_to.value = suspend_to
        with track_caller():
            jump(self.state.value)

        label(suspend_to)
        self.state.value = global_resume_to.value


def create_static_task[R](
    coro: Awaitable[R], on_return: Callable[[R], None] | None = None
) -> StaticTask:
    # @functools.wraps(create_static_task)
    async def inner() -> R:
        return await coro

    task_start = mk_internal_label("task_start")

    with if_(False):
        label(task_start)
        ans = _static_coroutine(inner, _inline=True)._handler()
        exit_program()
        if on_return is not None:
            on_return(ans)
        unreachable_checked()

    return StaticTask(State(task_start))


def _suspend():
    suspend_cont = mk_internal_label("suspend_cont")

    global_resume_to.value = suspend_cont
    jump(global_suspend_to.value)
    label(suspend_cont)


def static_coroutine[**P, R](fn: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
    fn_wrapped = _static_coroutine(fn)

    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        return await fn_wrapped(*args, **kwargs)

    return inner


def _static_coroutine[**P, R](
    fn: Callable[P, Awaitable[R]], *, _inline: bool = False
) -> StaticCoroFn[Callable[P, R]]:
    # @subr
    @functools.wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        g = fn(*args, **kwargs).__await__()

        try:
            while True:
                item = next(g)
                if isinstance(item, CoroYield):
                    _suspend()
                elif isinstance(item, StaticCoro):
                    item.handler(*item.args, **item.kwargs)
                else:
                    raise TypeError(item)

        except StopIteration as e:
            return e.value

    return StaticCoroFn(inner if _inline else subr(inner))
