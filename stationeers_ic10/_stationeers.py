from enum import Enum


class BatchMode(Enum):
    MEAN = 0
    SUM = 1
    MIN = 2
    MAX = 3

    def as_literal(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return self.name
