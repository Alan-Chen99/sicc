from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeGuard, final

from ordered_set import OrderedSet


@final
class _empty_t:
    pass


_empty = _empty_t()


def cast_unchecked(x: Any) -> Any:
    return x


@dataclass
class Cell[T]:
    _value: T | _empty_t = _empty

    @contextmanager
    def bind(self, val: T) -> Iterator[T]:
        old = self._value
        self._value = val
        try:
            yield val
        finally:
            self._value = old

    def get[D](self, default: D = None) -> T | D:
        if not isinstance(self._value, _empty_t):
            return self._value
        return default

    def set(self, val: T | _empty_t = _empty):
        self._value = val

    @property
    def value(self) -> T:
        assert not isinstance(self._value, _empty_t)
        return self._value

    @value.setter
    def value(self, val: T):
        self.set(val)


def narrow_unchecked[T](val: Any, typ: type[T]) -> TypeGuard[T]:
    return True


def disjoint_union[T](xs: OrderedSet[T], ys: OrderedSet[T]) -> OrderedSet[T]:
    ans = xs.union(ys)
    assert len(ans) == len(xs) + len(ys)
    return ans


def late_fn[**P, R](fn: Callable[[], Callable[P, R]]) -> Callable[P, R]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        return fn()(*args, **kwargs)

    return inner
