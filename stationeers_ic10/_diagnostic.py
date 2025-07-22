from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DebugInfo:
    # TODO

    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: Any) -> bool:
        return True


def debug_info() -> DebugInfo:
    return DebugInfo()
