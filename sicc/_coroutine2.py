"""
experimental sketch of stack-based coroutine
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Coroutine
from typing import TypeVar

from ._api import Int
from ._api import UserValue
from ._api import ValLabelLike
from ._api import Variable
from ._api import add_label_provenance
from ._api import jump
from ._api import remove_label_provenance
from ._api import transmute
from ._api import undef
from ._api import unreachable_checked
from ._control_flow import if_
from ._core import Label
from ._diagnostic import track_caller
from ._stack import Pointer
from ._state import State
from ._subr import subr
from ._tracing import label
from ._tracing import mk_internal_label
from ._tree_utils import TreeSpec
from ._tree_utils import optree_dataclass
from ._utils import Cell

# register_exclusion(__file__)


class CoroYield:
    def __await__(self):
        yield self


# # seperate function so that you get a warning if you dont await
# async def coro_yield() -> None:
#     await CoroYield()


F_co = TypeVar("F_co", covariant=True)


# @optree_dataclass(frozen=True)
# class CoroState:
#     cont_to: Label


# class CoroController:
#     def step(self, ptr: Pointer[CoroState]):
#         pass


async def register_state[S](s: TreeSpec[S]) -> Pointer[S]:
    assert False


# @dataclass
# class CoroFn[**P, R]:
#     _handler: Callable[P, R]

#     async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
#         return self._handler(*args, **kwargs)


# def stack_coro[R](fn: Awaitable[R]):
#     def inner() -> R:
#         g = fn.__await__()

#         try:
#             while True:
#                 item = next(g)
#                 if isinstance(item, CoroYield):
#                     _suspend()
#                 # elif isinstance(item, StaticCoro):
#                 #     item.handler(*item.args, **item.kwargs)
#                 else:
#                     raise TypeError(item)

#         except StopIteration as e:
#             return e.value

#     assert False


# @asynccontextmanager
# async def save_state(s: State) -> AsyncIterator[None]:
#     tree = _coro_push(s.value)
#     yield
#     s.value = _coro_pop(tree)


@dataclass
class _RawCoro[**P, R]:
    subr: Callable[P, R]
    _push_size: Cell[int]
    _pop_size: Cell[int]

    @property
    def push_size(self) -> int:
        return self._push_size.value

    @property
    def pop_size(self) -> int:
        return self._pop_size.value

    def trace(self, *args: P.args, **kwargs: P.kwargs) -> None:
        with if_(False):
            self.subr(*args, **kwargs)

    async def call(self, *args: P.args, **kwargs: P.kwargs) -> R:
        self.trace(*args, **kwargs)

        await _CoroPush(self.push_size)
        ans = self.subr(*args, **kwargs)
        await _CoroPop(self.pop_size)
        return ans


def _raw_coro[**P, R](fn: Callable[P, Awaitable[R]]) -> _RawCoro[P, R]:

    res_push_size: Cell[int] = Cell()
    res_pop_size: Cell[int] = Cell()

    @subr
    @functools.wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        g = fn(*args, **kwargs).__await__()
        push_size = 0
        offset = 0
        try:
            while True:
                item = next(g)
                if isinstance(item, _CoroPush):
                    offset += item.size
                elif isinstance(item, _CoroPop):
                    offset -= item.size
                else:
                    raise TypeError(item)
                push_size = max(push_size, offset)

        except StopIteration as e:
            ans = e.value

        res_push_size.value = push_size
        res_pop_size.value = push_size - offset

        return ans

    return _RawCoro(inner, res_push_size, res_pop_size)


# # @_raw_coro
# async def _return_trampoline() -> None:
#     # TODO: this cant be a trampoline yet, that messes with lifetime analysis
#     ra = await _coro_pop(TreeSpec.primitive(Label))
#     jump(ra)


def coroutine[**P, R](fn: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
    ret_state: State[R] = State()

    @_raw_coro
    async def fn_wrapped(ret_to: ValLabelLike, *args: P.args, **kwargs: P.kwargs) -> None:
        ret_to_, phantom_loc = remove_label_provenance(ret_to)
        await _coro_push(ret_to_)

        ans = await fn(*args, **kwargs)
        ret_state.value = ans

        ra = await _coro_pop(TreeSpec.primitive(Label))
        jump(add_label_provenance(ra, phantom_loc))

    @functools.wraps(fn)
    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        l = mk_internal_label("coro_call_ret")
        await fn_wrapped.call(l, *args, **kwargs)
        label(l)
        return ret_state.value

    return inner


# @dataclass
# class StaticCoroFn(Generic[F_co]):
#     _handler: F_co

#     def __call__[**P, R](
#         self: StaticCoroFn[Callable[P, R]], *args: P.args, **kwargs: P.kwargs
#     ) -> StaticCoro[R]:
#         args, kwargs = copy_tree((args, kwargs))
#         return StaticCoro(self._handler, args, kwargs)


# @dataclass
# class StaticCoro[R](Awaitable[R]):
#     handler: Callable[..., R]
#     args: Any
#     kwargs: Any

#     def __await__(self) -> Generator[Any, Any, R]:
#         with track_caller():
#             ans = yield self
#         return ans


cur_stack_ptr: Variable[int] = Variable(int)


# FIXME:
# currently, push and pop size tracking does not know control flow
# so one push must be paired with one pop


@dataclass
class _CoroPush:
    size: int

    def __await__(self):
        yield self


async def _coro_push[T](val: T) -> TreeSpec[T]:
    _, spec = TreeSpec.flatten(val)

    Pointer(_addr=cur_stack_ptr, _tree=spec)._push(val)
    await _CoroPush(len(spec))

    return spec


@dataclass
class _CoroPop:
    size: int

    def __await__(self):
        yield self


async def _coro_pop[T](spec: TreeSpec[T]) -> T:
    await _CoroPop(len(spec))
    return Pointer(_addr=cur_stack_ptr, _tree=spec)._pop()


cur_base_addr: Variable[int] = Variable(int)


def _cur_base_addr_ptr() -> Pointer[Int]:
    return Pointer(_addr=cur_base_addr.value, _tree=TreeSpec.primitive(int))


global_suspend_to: Variable[Label] = Variable(Label)

##########


@_raw_coro
async def _suspend_trampoline(cont_to: ValLabelLike) -> None:
    await _coro_push(cont_to)
    _cur_base_addr_ptr().write(cur_stack_ptr.value)
    jump(global_suspend_to.value)


async def suspend() -> None:
    with track_caller():
        cont_to = mk_internal_label("cont_to")
        await _suspend_trampoline.call(cont_to)
        unreachable_checked()

        await _CoroPop(1)
        label(cont_to)


##########


@subr
def _cont_trampoline(suspend_to: UserValue[Label], base_addr: Int) -> None:
    global_suspend_to.value = suspend_to
    cur_base_addr.value = base_addr
    cur_stack_ptr.value = _cur_base_addr_ptr().read()

    @_raw_coro
    async def _inner(base_addr: Int) -> None:
        cont_to = await _coro_pop(TreeSpec.primitive(Label))
        jump(cont_to)

    _inner.subr(base_addr)
    unreachable_checked()


def _coro_cont(base_addr: Int) -> None:
    suspend_to = mk_internal_label("suspend_to")

    _cont_trampoline(suspend_to, base_addr)
    label(suspend_to)


##########


@dataclass
class Task:
    buffer: Pointer[CoroBuffer]
    # _base_addr: Int

    def step(self) -> None:
        with track_caller():
            _coro_cont(self.buffer._addr)


@optree_dataclass(frozen=True, kw_only=True)
class CoroBuffer:
    _buffer: list[Any]
    # _wrapped: _RawCoro[[], None] = field(pytree_node=False)

    @staticmethod
    def create(size: int) -> CoroBuffer:
        return CoroBuffer(_buffer=[undef() for _ in range(size)])


# def create_state(coro: Awaitable[None]) -> CoroState:
#     @_raw_coro
#     async def inner() -> None:
#         await coro
#         unreachable_checked()

#     assert inner.push_size == inner.pop_size

#     return CoroState(
#         _buffer=[undef(AnyType) for _ in range(inner.push_size + 1)],
#         _wrapped=inner,
#     )


# def create_task(ptr: Pointer[CoroState]) -> Task:
#     inner = ptr.static_project(lambda s: s._wrapped)

#     assert False


# # @subr
# def _coro_init_trampoline(suspend_to: UserValue[Label], base_addr: Int, start_label: ValLabelLike):
#     global_suspend_to.value = split_lifetime(suspend_to)
#     cur_base_addr.value = base_addr
#     cur_stack_ptr.value = base_addr + 1
#     jump(start_label)


def create_task(coro: Awaitable[None], buffer: Pointer[CoroBuffer]) -> Task:
    @_raw_coro
    async def inner() -> None:
        await coro
        unreachable_checked()

    coro_init_end = mk_internal_label("coro_init_end")

    global_suspend_to.value = transmute(coro_init_end)
    cur_base_addr.value = buffer._addr
    cur_stack_ptr.value = buffer._addr + 1
    inner.subr()
    unreachable_checked()

    label(coro_init_end)

    assert inner.push_size == inner.pop_size
    buf_size = buffer.static_project(lambda b: len(b._buffer))
    if buf_size < inner.push_size + 1:
        raise RuntimeError(
            f"coroutine buffer not large enough: size is {buf_size}; "
            f"at least {inner.push_size + 1} required"
        )

    return Task(buffer)


# def static_coroutine[**P, R](fn: Callable[P, Awaitable[R]]) -> Callable[P, Coroutine[Any, Any, R]]:
#     fn_wrapped = _static_coroutine(fn)

#     async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
#         return await fn_wrapped(*args, **kwargs)

#     return inner


# def _static_coroutine[**P, R](
#     fn: Callable[P, Awaitable[R]], *, _inline: bool = False
# ) -> StaticCoroFn[Callable[P, R]]:
#     # @subr
#     @functools.wraps(fn)
#     def inner(*args: P.args, **kwargs: P.kwargs) -> R:
#         g = fn(*args, **kwargs).__await__()

#         try:
#             while True:
#                 item = next(g)
#                 if isinstance(item, CoroYield):
#                     _suspend()
#                 elif isinstance(item, StaticCoro):
#                     item.handler(*item.args, **item.kwargs)
#                 else:
#                     raise TypeError(item)

#         except StopIteration as e:
#             return e.value

#     return StaticCoroFn(inner if _inline else subr(inner))
