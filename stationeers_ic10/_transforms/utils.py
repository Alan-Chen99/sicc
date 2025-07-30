from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Concatenate

from rich import print as print

from .._core import Fragment
from .._tracing import internal_transform
from .._utils import ByIdMixin


@dataclass
class TransformCtx:
    frag: Fragment

    _cache: dict[CachedFn[..., Any], Any]


class Transform[**P, R](ByIdMixin):
    def __init__(self, fn: Callable[Concatenate[TransformCtx, P], R]) -> None:
        self.fn = fn
        self.id = id(fn)

    def __call__(self, frag: Fragment, /, *args: P.args, **kwargs: P.kwargs) -> R:
        from .basic import get_basic_index

        if not isinstance(self, CachedFn):
            before = copy.deepcopy(frag)
        else:
            before = None

        with internal_transform(frag):
            try:
                ctx = TransformCtx(frag, {})
                ans = self.fn(ctx, *args, **kwargs)

                if isinstance(self, CachedFn):
                    return ans

                assert before is not None
                if frag.blocks != before.blocks:
                    print("changed:", self.fn)
                else:
                    return ans

                # TODO: potentially reuse result of this call
                # validate result
                try:
                    get_basic_index(frag)
                except Exception as e:
                    raise RuntimeError(f"Transform {self.fn} returned invalid fragment") from e

            except:
                if before is not None:
                    print(before.__rich__(title=f"Fragment before {self.fn}"))
                print(frag.__rich__(title=f"Fragment after {self.fn}"))
                raise

            return ans


class CachedFn[**P, R](Transform[P, R]):
    def call_cached(self, ctx: TransformCtx, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if self in ctx._cache:
            return ctx._cache[self]
        ans = self.fn(ctx, *args, **kwargs)
        ctx._cache[self] = ans
        return ans


class LoopingTransform[**P](Transform[P, bool]):
    def __call__(self, frag: Fragment, /, *args: P.args, **kwargs: P.kwargs) -> bool:
        changed = False
        while super().__call__(frag, *args, **kwargs):
            changed = True
        return changed


def _run_phases_once(frag: Fragment, *fs: Callable[[Fragment], bool | None]) -> bool:
    changed = False

    for f in fs:
        ans = f(frag)
        if ans is not None:
            assert isinstance(ans, bool)
            changed = changed or ans

    return changed


def run_phases(frag: Fragment, *fs: Callable[[Fragment], bool | None], loop: bool = True) -> bool:
    if not loop:
        return _run_phases_once(frag, *fs)

    res = False
    while _run_phases_once(frag, *fs):
        res = True
    return res
