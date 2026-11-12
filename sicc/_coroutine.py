from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Coroutine
from typing import Generator
from typing import Generic
from typing import Never
from typing import TypeVar
from typing import cast

from ._api import ValLabelLike
from ._api import Variable
from ._api import jump
from ._api import unreachable_checked
from ._core import Label
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._state import State
from ._subr import FunctionRet
from ._subr import subr
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import trace_to_fragment
from ._tree_utils import copy_tree
from ._tree_utils import field as optree_field  # pyright: ignore[reportUnknownVariableType]
from ._tree_utils import optree_dataclass
from ._utils import allow_curry

register_exclusion(__file__)


class CoroYield:
    def __await__(self):
        yield self


# seperate function so that you get a warning if you dont await
async def coro_yield() -> None:
    await CoroYield()


F_co = TypeVar("F_co", covariant=True)


@dataclass
class StaticCoroFn(Generic[F_co]):
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


@optree_dataclass
class StaticTask:
    state: State[ValLabelLike] = optree_field(pytree_node=False)

    def step(self) -> None:
        suspend_to = mk_internal_label("suspend_to")

        global_suspend_to.value = suspend_to
        with track_caller():
            jump(self.state.value)

        label(suspend_to)
        self.state.value = global_resume_to.value


def create_static_task(coro: Awaitable[FunctionRet[Never]], /) -> StaticTask:

    # @functools.wraps(coro)
    async def inner():
        return await coro

    task_start = mk_internal_label("task_start")

    with trace_to_fragment(emit=True):
        label(task_start)
        _static_coroutine(inner, inline=True)._handler()
        unreachable_checked()

    return StaticTask(State(task_start))


def _suspend():
    suspend_cont = mk_internal_label("suspend_cont")

    global_resume_to.value = suspend_cont
    jump(global_suspend_to.value)
    label(suspend_cont)


@allow_curry
def static_coroutine[**P, R](
    fn: Callable[P, Awaitable[R]], /, *, inline: bool = False
) -> Callable[P, Coroutine[Any, Any, R]]:
    fn_wrapped = _static_coroutine(fn, inline=inline)

    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        return await fn_wrapped(*args, **kwargs)

    return inner


def _static_coroutine[**P, R](
    fn: Callable[P, Awaitable[R]], *, inline: bool = False
) -> StaticCoroFn[Callable[P, R]]:

    @subr(inline=inline)
    @functools.wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        g = fn(*args, **kwargs).__await__()

        try:
            sent: Any = None
            while True:
                item = g.send(sent)
                if isinstance(item, CoroYield):
                    _suspend()
                    sent = None
                elif isinstance(item, StaticCoro):
                    coro = cast(StaticCoro[Any], item)
                    sent = coro.handler(*coro.args, **coro.kwargs)
                else:
                    raise TypeError(item)

        except StopIteration as e:
            return e.value

    return StaticCoroFn(inner)
