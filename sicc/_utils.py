from __future__ import annotations

import importlib.util
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Final
from typing import Iterator
from typing import Protocol
from typing import TypeGuard
from typing import cast
from typing import final
from weakref import WeakSet

from ordered_set import OrderedSet

################################################################################


@final
class empty_t:
    pass


empty = empty_t()


################################################################################


@dataclass
class Cell[T]:
    _value: T | empty_t = empty

    @contextmanager
    def bind(self, val: T) -> Iterator[T]:
        old = self._value
        self._value = val
        try:
            yield val
        finally:
            self._value = old

    @contextmanager
    def bind_clear(self) -> Iterator[None]:
        old = self._value
        self._value = empty
        try:
            yield None
        finally:
            self._value = old

    def get[D](self, default: D = None) -> T | D:
        if not isinstance(self._value, empty_t):
            return self._value
        return default

    def set(self, val: T | empty_t = empty):
        self._value = val

    @property
    def value(self) -> T:
        assert not isinstance(self._value, empty_t)
        return self._value

    @property
    def __call__(self) -> T:
        return self.value

    @value.setter
    def value(self, val: T):
        self.set(val)

    def __bool__(self) -> bool:
        return bool(self.get(False))


################################################################################


def cast_unchecked(x: Any) -> Any:
    return x


def cast_unchecked_val[T](_v: T) -> Callable[[Any], T]:
    def inner(x: Any) -> Any:
        return x

    return inner


def safe_cast[T](_typ: type[T], x: T) -> T:
    return x


def narrow_unchecked[T](val: Any, typ: type[T]) -> TypeGuard[T]:
    return True


def isinst[T](val: Any, typ: type[T]) -> TypeGuard[T]:
    # narrow that permit typ being a generic
    return isinstance(val, typ)


def late_fn[F](fn: Callable[[], F]) -> F:
    def inner(*args: Any, **kwargs: Any) -> Any:
        return cast_unchecked(fn())(*args, **kwargs)

    return cast_unchecked(inner)


def in_typed[T](x: T, s: set[T] | list[T] | OrderedSet[T] | WeakSet[T]) -> bool:
    return x in s


def is_eq_typed[T](x: T) -> Callable[[T], bool]:
    def inner(y: T) -> bool:
        return x == y

    return inner


def mk_ordered_set() -> OrderedSet[Any]:  # makes type check happy
    return OrderedSet(())


@dataclass
class ReprAs:
    x: str

    def __repr__(self) -> str:
        return self.x


################################################################################


def disjoint_union[T](xs: OrderedSet[T], ys: OrderedSet[T]) -> OrderedSet[T]:
    ans = xs.union(ys)
    if not len(ans) == len(xs) + len(ys):
        raise ValueError(f"expected disjoint, but {xs & ys} are in both")
    return ans


def crc32(s: str) -> int:
    val = zlib.crc32(s.encode())
    if val >= 2**31:
        val -= 2**32
    return val


################################################################################

counter = Cell(0)


def get_id() -> int:
    ans = counter.value
    counter.value = ans + 1
    return ans


_IdType = int | str | tuple[int, ...]

_TYPE_TO_IDX: dict[type[_IdType], int] = {
    int: 1,
    str: 2,
    tuple: 3,
}


class _HaveId(Protocol):
    id: Final[_IdType]


class ByIdMixin:
    """
    child that is a dataclass should set eq=False for the __eq__ here to take effect
    """

    def __eq__(self: _HaveId, other: object) -> bool:
        return type(other) is type(self) and other.id == self.id

    def __hash__(self: _HaveId) -> int:
        return hash(self.id)

    def __lt__(self, other: _HaveId) -> bool:
        id1 = cast(_HaveId, self).id
        id2 = other.id
        return (_TYPE_TO_IDX[type(id1)], id1) < (_TYPE_TO_IDX[type(id2)], id2)


class Singleton(ByIdMixin):
    def __init__(self) -> None:
        self.id = type(self).__module__ + "." + type(self).__qualname__

    def __repr__(self) -> str:
        return type(self).__name__


################################################################################


def load_module_from_file(file_path: Path, module_name: str) -> ModuleType:
    if not file_path.exists():
        raise ValueError(f"file does not exist: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    # otherwise @dataclass dont work
    sys.modules[module_name] = module

    prev_path = sys.path.copy()
    try:
        sys.path.insert(0, str(file_path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path = prev_path
    return module
