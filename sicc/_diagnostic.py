from __future__ import annotations

import contextlib
import gc
import linecache
import logging
import os
import sys
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from types import ModuleType
from types import TracebackType
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Never
from typing import Self

import rich
import rich.repr
from rich import print as print  # autoflake: skip
from rich.console import Group
from rich.console import RenderableType
from rich.console import group
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Frame
from rich.traceback import Stack
from rich.traceback import Trace
from rich.traceback import Traceback

from . import _utils
from ._utils import Cell
from .config import RICH_SPINNER
from .config import _status_text
from .config import show_src_info
from .config import verbose

if TYPE_CHECKING:
    from ._core import Fragment

TRACEBACK_SUPPRESS: Cell[set[str]] = Cell(set())


def register_exclusion(filename: str | ModuleType):
    # taken from rich; we want to "get the top frame" also, aside from printing
    if not isinstance(filename, str):
        f = filename.__file__
        assert f is not None
        f = os.path.dirname(f)
    else:
        f = filename
    TRACEBACK_SUPPRESS.value.add(f)


register_exclusion(__file__)
register_exclusion(_utils.__file__)

register_exclusion(contextlib)


class Warnings(Enum):
    InternalError = "InternalError"
    Unspecified = "Unspecified"
    Unused = "Unused"


class SuppressExit(Exception):
    """
    exception indicating that caller should exit with code without showing a backtrace
    """

    def __init__(self, code: int):
        self.code = code


def get_location(depth: int = 0) -> Frame:
    frame = sys._getframe(depth + 1)
    while True:
        if frame.f_back is None:
            break
        if not any(frame.f_code.co_filename.startswith(x) for x in TRACEBACK_SUPPRESS.value):
            break
        frame = frame.f_back

    ex = BaseException("_tmp")
    ans = Traceback.extract(
        type(ex),
        ex,
        TracebackType(None, frame, frame.f_lasti, frame.f_lineno),
    )
    (stack,) = ans.stacks
    (ans_frame,) = stack.frames
    return ans_frame


def format_location(loc: Frame):
    stack = Stack("", "", frames=[loc])
    return Traceback(Trace([stack]), suppress=TRACEBACK_SUPPRESS.value)._render_stack(stack)


def get_trace(depth: int = 0) -> Trace:
    # frame = _get_traceback_frame()
    frame = sys._getframe(depth + 1)

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
    ans = Traceback.extract(type(ex), ex, tb)
    (stack,) = ans.stacks
    stack.exc_type = ""
    stack.exc_value = ""
    return ans


def format_backtrace(trace: Trace, max_frames: int = 100):
    stack = Stack("", "", frames=trace.stacks[0].frames[-max_frames:])
    return Traceback(trace)._render_stack(stack)


def frame_short_desc(frame: Frame) -> str:
    # print("get_callsite_desc", tb)
    # its not from an ex so it has no cause, "stacks" should always have len 1
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
            assert len(ans.debug._must_use_ctx) == 0
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
        yield self.debug.location_info_brief()

    def check(self):
        assert len(self.debug._must_use_ctx) == 0
        if len(self._get_parents()) == 0:
            self.debug.warn("expression has no effect:", typ=Warnings.Unused)


def check_must_use():
    gc.collect()
    for x in mustuse_ctxs.value:
        x.check()
    mustuse_ctxs.value = []


@contextmanager
def must_use() -> Iterator[None]:
    mc = MustuseCtx.new()
    tmp_di = DebugInfo(_must_use_ctx=[mc])
    mc.add_parent(tmp_di)

    with add_debug_info(tmp_di):
        del mc, tmp_di
        yield


@dataclass
class DebugInfo:
    created_at: str | None = None
    traceback: Trace | None = None
    # other files should use `fuse_must_use`
    _must_use_ctx: list[MustuseCtx] = field(default_factory=lambda: [])
    describe: str = ""
    location: Frame | None = None
    track_caller: bool = False
    suppress_warnings: set[Warnings] = field(default_factory=lambda: set())

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
        yield "must_use_ctx", self._must_use_ctx
        yield "describe", self.describe

    def fuse_must_use(self, other: DebugInfo) -> Self:
        self._must_use_ctx += other._must_use_ctx
        for x in other._must_use_ctx:
            x.add_parent(self)
        return self

    def fuse(self, other: DebugInfo) -> Self:
        """
        other takes priority
        modifies self, leaves other untouched
        """
        self.traceback = other.traceback or self.traceback
        self.fuse_must_use(other)
        self.describe = other.describe or self.describe
        self.location = other.location or self.location
        self.track_caller = other.track_caller or self.track_caller
        self.suppress_warnings |= other.suppress_warnings
        return self

    def location_info_brief(self) -> str:
        if not show_src_info.value:
            return ""

        if desc := self.describe:
            return desc
        # if loc := self.location:
        #     return frame_short_desc(loc)
        if (verbose.value >= 3 or self.track_caller) and (loc := self.location):
            return frame_short_desc(loc)
        return ""

    @group()
    def location_info_full(self) -> Iterator[RenderableType]:
        if desc := self.describe:
            yield f"({desc})"
        # if tb := self.traceback:
        #     yield format_backtrace(tb)
        #     return
        if loc := self.location:
            yield format_location(loc)

    def __rich__(self) -> RenderableType:
        return self.location_info_full()

    def warn(self, msg: str, typ: Warnings = Warnings.Unspecified) -> Report:
        return mk_warn(msg, self, typ=typ, suppress=typ in self.suppress_warnings)

    def error(self, msg: str, typ: Warnings = Warnings.Unspecified) -> Report:
        return mk_error(msg, self, typ=typ, suppress=typ in self.suppress_warnings)


_PENDING_ERRORS: Cell[list[Report]] = Cell([])


def show_pending_diagnostics():
    pending = _PENDING_ERRORS.value
    _PENDING_ERRORS.value = []
    for x in pending:
        x.show()


def mk_warn(
    msg: str | Text,
    first: RenderableType | None = None,
    *parts: RenderableType,
    typ: Warnings = Warnings.Unspecified,
    suppress: bool = False,
) -> Report:
    txt = Text("", style="logging.level.warning")
    txt += Text("[WARN] ", style="bold")
    txt += msg
    return Report.new(
        Group(txt, first) if first is not None else txt,
        *parts,
        border_style="logging.level.warning",
        typ=typ,
        suppress=suppress,
    )


def mk_error(
    msg: str | Text,
    first: RenderableType | None = None,
    *parts: RenderableType,
    suppress: bool = False,
    typ: Warnings = Warnings.Unspecified,
) -> Report:
    txt = Text("", style="logging.level.error")
    txt += Text("[ERROR] ", style="bold")
    txt += msg
    return Report.new(
        Group(txt, first) if first is not None else txt,
        *parts,
        border_style="logging.level.error",
        typ=typ,
        suppress=suppress,
    )


@dataclass
class Report:
    # internal part of bt (bt of line that created the report)
    typ: Warnings
    suppress: bool
    trace: Trace | None
    parts: list[RenderableType]

    border_style: str = "traceback.border"
    did_show: bool = False

    @staticmethod
    def new(
        *parts: RenderableType,
        border_style: str = "traceback.border",
        depth: int = 0,
        typ: Warnings = Warnings.Unspecified,
        suppress: bool = False,
    ) -> Report:
        logging.info(f"created diagnostic report: {_status_text()}")
        ans = Report(
            typ=typ,
            suppress=suppress,
            trace=get_trace(depth + 1) if verbose.value >= 1 else None,
            border_style=border_style,
            parts=list(parts),
        )
        _PENDING_ERRORS.value.append(ans)
        return ans

    @staticmethod
    def from_ex(e: BaseException) -> Report:
        txt = Text("internal compiler error:", style="logging.level.error")
        tb = Traceback.from_exception(type(e), e, e.__traceback__)
        return Report(Warnings.InternalError, False, get_trace(), [Group(txt, tb)])

    def add(self, *parts: RenderableType) -> Self:
        self.parts.append(Group(*parts))
        return self

    def note(self, msg: str, *parts: RenderableType) -> Self:
        txt = Text()
        txt.append("[NOTE] ", style="traceback.note")
        self.add(txt + Text(msg), *parts)
        return self

    def throw(self) -> Never:
        raise CompilerError(self)

    def fatal(self) -> Never:
        show_pending_diagnostics()
        self.show()
        raise SuppressExit(1) from None

    def __rich__(self) -> RenderableType:
        def gen():
            yield self.parts[0]
            for x in self.parts[1:]:
                yield ""
                yield x
            if self.trace is not None:
                yield ""
                yield Text("Traceback (most recent call last)", style="traceback.title")
                yield format_backtrace(self.trace)

        return Panel(Group(*gen()), border_style=self.border_style)

    def show(self) -> None:
        if self.did_show:
            return
        if self.suppress:
            return
        if status := RICH_SPINNER.get():
            status.stop()
            try:
                print(self)
            finally:
                status.start()
        else:
            print(self)
        self.did_show = True


class CompilerError(Exception):
    def __init__(self, report: Report):
        self.report = report


@contextmanager
def catch_ex_and_exit(f: Fragment | None = None) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        if isinstance(e, CompilerError):
            report = e.report
        else:
            report = Report.from_ex(e)
        if verbose.value >= 1:
            if f is not None:
                report.add(f.__rich__(title=f"While handling fragment:"))
        report.fatal()


def debug_info(depth: int = 1) -> DebugInfo:
    frame = sys._getframe(1)

    ans = DebugInfo(
        # created_at=linecache.getlines(frame.f_code.co_filename)[frame.f_lineno - 1].strip(),
        created_at=frame.f_code.co_qualname,
    )
    # print("stack:", DEBUG_INFO_STACK.value)
    for x in DEBUG_INFO_STACK.value:
        ans.fuse(x)
    if ans.traceback is None:
        ans.traceback = get_trace()
    if ans.location is None:
        ans.location = get_location(depth + 1)

    return ans


DEBUG_INFO_STACK: Cell[list[DebugInfo]] = Cell([])


@contextmanager
def clear_debug_info() -> Iterator[None]:
    with DEBUG_INFO_STACK.bind([]):
        yield None


@contextmanager
def add_debug_info(x: DebugInfo) -> Iterator[None]:
    # takes priority over stuff under
    with DEBUG_INFO_STACK.bind([x] + DEBUG_INFO_STACK.value):
        yield None


@contextmanager
def track_caller(depth: int = 0) -> Iterator[None]:
    # FIXME: this 3 is impl detail of contextlib
    # TODO: improve efficiency by checking if already in "track_caller"
    # so this would do nothing
    loc = get_location(depth + 3)
    tb = get_trace()
    with add_debug_info(DebugInfo(location=loc, traceback=tb, track_caller=True)):
        yield


@contextmanager
def suppress_warnings(*warnings: Warnings) -> Iterator[None]:
    with add_debug_info(DebugInfo(suppress_warnings=set(warnings))):
        yield


def describe_fn(fn: Any) -> str:
    try:
        return fn.__module__ + "." + fn.__qualname__
    except AttributeError:
        return repr(fn)
