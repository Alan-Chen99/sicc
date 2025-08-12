from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Concatenate

from rich import print as print  # autoflake: skip
from rich.pretty import pretty_repr

from .._core import Fragment
from .._diagnostic import CompilerError
from .._diagnostic import Report
from .._diagnostic import register_exclusion
from .._tracing import internal_transform
from .._utils import Cell
from ..config import verbose
from ..config import with_status

register_exclusion(__file__)


@dataclass
class TransformCtx:
    frag: Fragment

    _cache: dict[Any, Any]


frag_is_global: Cell[bool] = Cell(False)


class Transform[**P, R]:
    def __init__(self, fn: Callable[Concatenate[TransformCtx, P], R]) -> None:
        self.fn = fn
        self.sig = inspect.signature(fn)
        self.id = id(fn)

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(id(self.fn))

    def __rich_repr__(self):
        yield self.fn.__name__

    def __call__(self, frag: Fragment, /, *args: P.args, **kwargs: P.kwargs) -> R:
        from .basic import get_index

        if not isinstance(self, CachedFn):
            before = copy.deepcopy(frag)
        else:
            before = None

        with internal_transform(frag), with_status(self.fn.__qualname__):
            try:
                logging.debug(f"running {self.fn.__qualname__}")
                ctx = TransformCtx(frag, {})
                ans = self.fn(ctx, *args, **kwargs)

                if isinstance(self, CachedFn):
                    return ans

                assert before is not None
                if frag.blocks != before.blocks:
                    logging.info(f"modified by {self.fn.__qualname__}")

                # TODO: potentially reuse result of this call
                # validate result
                try:
                    get_index(frag)
                except Exception as e:
                    raise RuntimeError(f"Transform {self.fn} returned invalid fragment") from e

                return ans
            except Exception as e:
                if isinstance(e, CompilerError):
                    report = e.report
                else:
                    report = Report.from_ex(e)
                report.note(f"during transform {self.fn.__qualname__}")
                if verbose.value >= 1:
                    if before is not None:
                        report.add(
                            before.__rich__(title=f"Fragment before {self.fn}"),
                        )
                    if before is None or frag.blocks != before.blocks:
                        report.add(
                            frag.__rich__(title=f"Fragment after {self.fn}"),
                        )
                report.fatal()


class CachedFn[**P, R](Transform[P, R]):
    def call_cached(self, ctx: TransformCtx, /, *args: P.args, **kwargs: P.kwargs) -> R:
        # TODO: make this cache default arguments friendly

        bound = self.sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        cache_key = (self, *sorted(bound.arguments.items()))

        if cache_key in ctx._cache:
            logging.debug(f"cached: {pretty_repr(cache_key)}")
            return ctx._cache[cache_key]
        logging.debug(f"cach miss: {pretty_repr(cache_key)}")
        ans = self.fn(ctx, *args, **kwargs)
        ctx._cache[cache_key] = ans
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
