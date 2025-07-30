from __future__ import annotations

import linecache
import sys
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import FrameType
from types import TracebackType
from typing import Any
from typing import Iterator
from typing import Self

import rich
import rich.repr
from rich import print as print
from rich.traceback import Trace
from rich.traceback import Traceback

from ._utils import Cell
from ._utils import register_exclusion_

_TRACEBACK_EXCLUDE_MODULES: Cell[set[str]] = Cell({f"contextlib"})
_TRACEBACK_EXCLUDE_FILES: Cell[set[str]] = Cell(set())


def register_exclusion(filename: str):
    _TRACEBACK_EXCLUDE_FILES.value.add(filename)


register_exclusion(__file__)
register_exclusion_()


def _get_traceback_frame() -> FrameType:
    frame = sys._getframe()
    while True:
        if frame.f_back is None:
            return frame
        if (
            frame.f_code.co_filename not in _TRACEBACK_EXCLUDE_FILES.value
            and frame.f_globals.get("__name__") not in _TRACEBACK_EXCLUDE_MODULES.value
        ):
            return frame
        frame = frame.f_back


def get_trace() -> Trace:
    frame = _get_traceback_frame()
    # print("_get_traceback_frame", frame)

    # note: "frame" is mutable so we cant do this function lazily

    # see
    # https://github.com/Textualize/rich/discussions/1531
    tb = None
    while True:
        tb = TracebackType(tb, frame, frame.f_lasti, frame.f_lineno)
        frame = frame.f_back
        if frame is None:
            break

    ex = BaseException("_tmp")
    return Traceback.extract(type(ex), ex, tb)


@dataclass
class CallInfo:
    filename: str


def _get_callsite_desc(tb: Trace) -> str:
    # print("get_callsite_desc", tb)
    # its not from an ex so it has no cause, "stacks" should always have len 1
    frame = tb.stacks[0].frames[-1]
    if frame.last_instruction is None:
        return ""
    (start_line, start_column), (end_line, end_column) = frame.last_instruction
    if start_line == end_line:
        code_lines = linecache.getlines(frame.filename)
        return (
            Path(frame.filename).name
            + ":"
            + str(start_line)
            + ": "
            + code_lines[start_line - 1][start_column:end_column]
        )
    return Path(frame.filename).name + ":" + str(start_line) + ".." + str(end_line)


mustuse_ctxs: Cell[list[MustuseCtx]] = Cell([])


@dataclass
class MustuseCtx:
    parents: list[weakref.ref[DebugInfo]]
    debug: DebugInfo

    @staticmethod
    def new() -> MustuseCtx:
        with track_caller():
            ans = MustuseCtx([], debug_info())
            assert len(ans.debug.must_use_ctx) == 0
            mustuse_ctxs.value.append(ans)
            return ans

    def add_parent(self, parent: DebugInfo):
        self.parents.append(weakref.ref(parent))

    def _get_parents(self):
        return [val for x in self.parents if (val := x())]

    def __rich_repr__(self) -> rich.repr.Result:
        # yield "parents", [(x.created_at, x.describe) for x in self._get_parents()]
        # yield "parents", [[type(x) for x in gc.get_referrers(x)] for x in self._get_parents()]

        # yield len(gc.get_referrers(self))
        # yield [type(x) for x in gc.get_referrers(self)]
        # yield [type(x) for x in gc.get_referrers(self)]
        yield id(self)
        yield self.debug.location_info()

    def check(self):
        assert len(self.debug.must_use_ctx) == 0
        if len(self._get_parents()) == 0:
            print("WARN: UNUSED:", self)
            # print([(x.describe, gc.get_referrers(x)) for x in self._get_parents()])

        # if len(self._get_parents()) == 0:
        #     print("WARN: UNUSED:", self.debug.describe)


def check_must_use():
    for x in mustuse_ctxs.value:
        x.check()


@contextmanager
def must_use() -> Iterator[None]:
    mc = MustuseCtx.new()
    tmp_di = DebugInfo(must_use_ctx=[mc])
    mc.add_parent(tmp_di)

    with add_debug_info(tmp_di):
        del mc, tmp_di
        yield


@dataclass
class DebugInfo:
    created_at: str | None = None
    traceback: Trace | None = None
    must_use_ctx: list[MustuseCtx] = field(default_factory=lambda: [])
    describe: str = ""

    def __eq__(self, other: Any) -> bool:
        # if we get here the caller probably didnt define a __eq__
        # which is probably a bug, so we dont just return True
        raise RuntimeError("unreachable")

    def __hash__(self) -> int:
        raise RuntimeError("unreachable")

    def __deepcopy__(self, memo: Any):
        # TODO: this is fine since we are only ever using deep copy
        # for checking in _transforms/utils.py
        # we may want to clone "mustuse" trackers
        return self

    def __rich_repr__(self) -> rich.repr.Result:
        if self.created_at is not None:
            yield "created_at", self.created_at
        yield "have_tb", (self.traceback is not None)
        yield "must_use_ctx", self.must_use_ctx
        yield "describe", self.describe

    def fuse(self, other: DebugInfo) -> Self:
        """
        other takes priority
        modifies self, leaves other untouched
        """
        self.traceback = other.traceback or self.traceback
        self.must_use_ctx += other.must_use_ctx
        for x in other.must_use_ctx:
            x.add_parent(self)
        self.describe = other.describe or self.describe
        return self

    def location_info(self) -> str:
        return self.describe
        # if not self.describe:
        #     return ""
        # if self.traceback is None:
        #     return ""
        # return _get_callsite_desc(self.traceback)

    def error(self, msg: str) -> Report:
        raise RuntimeError(msg)


@dataclass
class Report:
    parts: list[Any]

    def note(self, msg: str, loc: DebugInfo) -> Self:
        return self


def debug_info() -> DebugInfo:
    frame = sys._getframe(1)

    ans = DebugInfo(
        # created_at=linecache.getlines(frame.f_code.co_filename)[frame.f_lineno - 1].strip(),
        created_at=frame.f_code.co_qualname,
    )
    for x in DEBUG_INFO_STACK.value:
        ans.fuse(x)
    if ans.traceback is None:
        ans.traceback = get_trace()

    return ans


DEBUG_INFO_STACK: Cell[list[DebugInfo]] = Cell([])


@contextmanager
def add_debug_info(x: DebugInfo) -> Iterator[None]:
    # takes priority over stuff under
    with DEBUG_INFO_STACK.bind([x] + DEBUG_INFO_STACK.value):
        yield None


@contextmanager
def track_caller() -> Iterator[None]:
    tb = get_trace()
    with add_debug_info(DebugInfo(describe=_get_callsite_desc(tb), traceback=tb)):
        yield


def describe_fn(fn: Any):
    return fn.__module__ + "." + fn.__qualname__
