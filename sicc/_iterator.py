from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
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
from ._control_flow import break_
from ._control_flow import if_
from ._control_flow import loop
from ._control_flow import while_
from ._core import Label
from ._state import State
from ._tracing import label
from ._tracing import mk_internal_label
from ._utils import get_id


@dataclass(frozen=True)
class Iterator_[T]:
    # jump here to start
    it_start: Label

    # jump here to advance iterator
    it_resume: Variable[Label]

    # iterator jumps here to indicate item being yielded
    it_yield: Variable[Label]

    # iterator jumps here to indicate iterator finished
    it_end: Variable[Label]

    # value of next item, valid after iterator jumps to it_back
    it_value: State[T]

    @staticmethod
    def from_iterable[T1](it: Iterable[T1]) -> Iterator_[T1]:
        id = get_id()
        it_start = mk_internal_label("it_start", id)

        it_resume = Variable(Label)
        it_yield = Variable(Label)
        it_end = Variable(Label)
        it_value: State[T1] = State()

        with if_(False):
            label(it_start)

            for idx, item in enumerate(it):
                resume_to = mk_internal_label(f"it_resume({idx})", id)

                it_value.value = item
                it_resume.value = resume_to
                jump(it_yield)

                label(resume_to)

            jump(it_end)

        return Iterator_(
            it_start=it_start,
            it_resume=it_resume,
            it_yield=it_yield,
            it_end=it_end,
            it_value=it_value,
        )

    def __iter__(self) -> Iterator[T]:
        id = get_id()

        it_yield = mk_internal_label("iter_handle_item", id)
        it_end = mk_internal_label("iter_end", id)

        # start
        self.it_yield.value = it_yield
        self.it_end.value = it_end
        jump(self.it_start)

        # main loop (handle item)
        label(it_yield)
        value = self.it_value.value

        with (
            BREAK_TO.bind(it_end),
            CONTINUE_TO.bind(self.it_resume),
            ELSE_HOOK.bind_clear(),
        ):
            yield value

        jump(self.it_resume)

        # end
        label(it_end)


def wrap_generator[**P, T](fn: Callable[P, Iterator[T]]) -> Callable[P, Iterator_[T]]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> Iterator_[T]:
        return Iterator_.from_iterable(fn(*args, **kwargs))

    return inner


@overload
def range_(stop: Int, /) -> Iterator[VarRead[int]]: ...
@overload
def range_(start: Int, stop: Int, /) -> Iterator[VarRead[int]]: ...


@wrap_generator
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
