from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Iterator
from typing import Self

from ._utils import Cell


class Err(Enum):
    pass


@dataclass
class DebugInfo:
    # TODO

    def __eq__(self, other: Any) -> bool:
        # if we get here the caller probably didnt define a __eq__
        # which is probably a bug, so we dont just return True
        raise RuntimeError("unreachable")

    def __hash__(self) -> int:
        raise RuntimeError("unreachable")

    def mark_unused(self):
        pass

    def error(self, msg: str) -> Report:
        raise RuntimeError(msg)

    def fuse(self, other: DebugInfo) -> None:
        pass


@dataclass
class Report:
    parts: list[Any]

    def note(self, msg: str, loc: DebugInfo) -> Self:
        return self


def debug_info() -> DebugInfo:
    return DebugInfo()


_DEBUG_INFO_OVERRIDE: Cell[DebugInfo] = Cell()


@contextmanager
def override_debug_info(x: DebugInfo) -> Iterator[None]:
    with _DEBUG_INFO_OVERRIDE.bind(x):
        yield None
