from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Callable
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import overload

from ._api import Int
from ._api import Variable
from ._api import VarRead
from ._api import jump
from ._control_flow import BREAK_TO
from ._control_flow import CONTINUE_TO
from ._control_flow import ELSE_HOOK
from ._control_flow import block
from ._control_flow import break_
from ._control_flow import clear_control_flow_hooks
from ._control_flow import if_
from ._control_flow import loop
from ._control_flow import while_
from ._core import Label
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import describe_fn
from ._diagnostic import track_caller
from ._state import State
from ._subr import _RETURN_HOOK
from ._subr import Subr
from ._subr import subr
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import trace_to_fragment
from ._utils import CallDescMixin
from ._utils import get_id


def patch_for_loop[E](
    it: Iterable[E], *, break_to: Label | None = None, emit_break_label: bool = True
) -> Iterator[E]:
    """
    continue_() and break_() in `for x in patch_for_loop(it): ...` work correctly
    """

    id = get_id()

    if not emit_break_label and break_to is None:
        raise TypeError("patch_for_loop(..., emit_break_label=False) requires break_to=")

    for_end = break_to if break_to is not None else mk_internal_label("for_end", id)

    with track_caller():
        for idx, item in enumerate(it):
            for_cont = mk_internal_label(f"for_cont({idx})", id)
            with (
                CONTINUE_TO.bind(for_cont),
                BREAK_TO.bind(for_end),
                ELSE_HOOK.bind_clear(),
            ):
                yield item
            label(for_cont)

    if emit_break_label:
        label(for_end)


################################################################################


def runtime_loop[E](it: Iterable[E]) -> Iterator[E]:
    id = get_id()
    it_yield = mk_internal_label("it_yield", id)
    it_finish = mk_internal_label("it_finish", id)

    it_resume = Variable(Label)
    it_value: State[E] = State()

    # generator part
    with (
        block() as b,
        clear_control_flow_hooks(),
        _RETURN_HOOK.bind(b.break_),
    ):
        for idx, item in enumerate(it):
            resume_to = mk_internal_label(f"it_resume({idx})", id)

            it_value.value = item
            it_resume.value = resume_to
            jump(it_yield)

            label(resume_to)

        label(it_finish)

    # caller part
    with trace_to_fragment(emit=True):
        label(it_yield)
        with (
            BREAK_TO.bind(it_finish),
            CONTINUE_TO.bind(it_resume),
            ELSE_HOOK.bind_clear(),
        ):
            yield it_value.value
        jump(it_resume)


def inline_generator[**P, T](fn: Callable[P, Iterator[T]]) -> Callable[P, Iterator[T]]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> Iterator[T]:
        return runtime_loop(fn(*args, **kwargs))

    return inner


################################################################################


@dataclass(frozen=True)
class GeneratorSubr[**P, Y, R](CallDescMixin):
    subr: tuple[Subr[P, R]]  # tuple to satisfy pyright

    # jump here to advance iterator
    it_resume: Variable[Label]

    # iterator jumps here to indicate item being yielded
    it_yield: Variable[Label]

    # value of next item, valid after iterator jumps to it_yield
    yield_value: State[Y]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Generator[Y, None, R]:
        (subr,) = self.subr

        id = get_id()
        yield_to = mk_internal_label("it_call_yield", id)
        it_finish = mk_internal_label("it_finish", id)

        self.it_yield.value = yield_to

        # trace subr first since that feels more sensible
        # same thing if no crash
        with track_caller():
            ans = subr(*args, **kwargs)

        label(it_finish)

        with trace_to_fragment(emit=True):
            with track_caller():
                label(yield_to)
                value = self.yield_value.value

            with (
                BREAK_TO.bind(it_finish),  # breaking causes return value to be undefined
                CONTINUE_TO.bind(self.it_resume),
                ELSE_HOOK.bind_clear(),
            ):
                yield value

            with add_debug_info(DebugInfo(describe=f"continuing {describe_fn(subr.fn)}")):
                jump(self.it_resume)

        return ans


def shared_generator[**P, Y, R](fn: Callable[P, Generator[Y, None, R]]) -> GeneratorSubr[P, Y, R]:
    it_resume = Variable(Label)
    it_yield = Variable(Label)

    yield_value: State[Y] = State()

    @subr
    @functools.wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        id = get_id()

        it = fn(*args, **kwargs)

        try:
            idx = 0
            while True:
                resume_to = mk_internal_label(f"it_resume({idx})", id)
                idx += 1

                val = next(it)

                with add_debug_info(DebugInfo(describe=f"yielding from {describe_fn(fn)}")):
                    yield_value.value = val
                    it_resume.value = resume_to
                    # jump to caller; caller will jump back to it_resume
                    jump(it_yield)

                # caller jumps back here; we continue to yield next value
                # with add_debug_info(DebugInfo(describe=f"continuing {describe_fn(fn)}")):
                # with track_caller():
                label(resume_to)

        except StopIteration as e:
            return e.value

    return GeneratorSubr(
        subr=(inner,),
        it_resume=it_resume,
        it_yield=it_yield,
        yield_value=yield_value,
    )


################################################################################


@overload
def range_(stop: Int, /) -> Iterator[VarRead[int]]: ...
@overload
def range_(start: Int, stop: Int, /) -> Iterator[VarRead[int]]: ...


@inline_generator
def range_(x: Int, y: Int | None = None, /) -> Iterator[VarRead[int]]:
    # will be implemented to support all args to python range in future
    if y is None:
        start, stop = 0, x
    else:
        start, stop = x, y

    if isinstance(start, int) and isinstance(stop, int) and start < stop:
        # do while is 1 less instruction
        idx = Variable(int, start)
        # some VarRead is defferred; make sure its only read once
        stop_ = Variable(int, stop)
        with loop():
            yield idx.value
            idx.value += 1
            with if_(idx.value == stop_):
                break_()
    else:
        idx = Variable(int, start)
        stop_ = Variable(int, stop)
        with while_(lambda: idx < stop):
            yield idx.value
            idx.value += 1
