from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Callable
from typing import Concatenate
from weakref import WeakKeyDictionary

from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.pretty import pretty_repr

from .._core import Block
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

#: analysis cache of the last transform on each fragment, with the fragment state it describes.
#: reused by the next transform only while that state is unchanged.
_ANALYSIS_CACHE: WeakKeyDictionary[Fragment, tuple[object, dict[Any, Any]]] = WeakKeyDictionary()


def _analysis_state(frag: Fragment) -> object:
    """everything cached analyses read from outside their arguments"""
    return (
        frag_is_global.value,
        tuple(frag.scope.private_labels),
        tuple(tuple(b.contents) for b in frag.blocks.values()),
    )


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
            # BoundInstr and Label are frozen and compare by id, so copying the containers is
            # enough for the change check below; rendering reads only private_labels from scope
            before = Fragment(
                frag.finished_init,
                {l: Block(list(b.contents), b.debug) for l, b in frag.blocks.items()},
                replace(frag.scope, private_labels=OrderedSet(frag.scope.private_labels)),
            )
        else:
            before = None

        state = _analysis_state(frag)
        prev = _ANALYSIS_CACHE.get(frag)
        cache = prev[1] if prev is not None and prev[0] == state else {}

        with internal_transform(frag), with_status(self.fn.__qualname__):
            try:
                logging.debug(f"running {self.fn.__qualname__}")
                ctx = TransformCtx(frag, cache)
                ans = self.fn(ctx, *args, **kwargs)

                if _analysis_state(frag) == state:
                    if isinstance(self, CachedFn):
                        # direct calls always run; the result still serves later lookups
                        cache[self.cache_key(*args, **kwargs)] = ans
                    _ANALYSIS_CACHE[frag] = (state, cache)
                else:
                    _ANALYSIS_CACHE.pop(frag, None)

                if isinstance(self, CachedFn):
                    return ans

                assert before is not None
                if frag.blocks != before.blocks:
                    logging.info(f"modified by {self.fn.__qualname__}")

                    if verbose.value >= 2:
                        print(
                            frag.__rich__(title=f"Fragment after {self.fn}"),
                        )

                    # validate result
                    try:
                        get_index(frag)
                    except CompilerError as e:
                        raise
                    except Exception as e:
                        raise RuntimeError(f"Transform {self.fn} returned invalid fragment") from e

                return ans
            except Exception as e:
                if isinstance(e, CompilerError):
                    if not e.add_transform_ctx:
                        raise e
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

                raise CompilerError(report, add_transform_ctx=False) from None


class CachedFn[**P, R](Transform[P, R]):
    def cache_key(self, *args: P.args, **kwargs: P.kwargs) -> tuple[Any, ...]:
        # None stands in for ctx, so positional, keyword and defaulted forms of a call agree
        bound = self.sig.bind_partial(None, *args, **kwargs)
        bound.apply_defaults()
        return (self, *sorted(bound.arguments.items()))

    def call_cached(self, ctx: TransformCtx, /, *args: P.args, **kwargs: P.kwargs) -> R:
        cache_key = self.cache_key(*args, **kwargs)

        # guarded: pretty_repr of the key is costly and this runs for every analysis lookup
        debug = logging.root.isEnabledFor(logging.DEBUG)
        if cache_key in ctx._cache:
            if debug:
                logging.debug(f"cached: {pretty_repr(cache_key)}")
            return ctx._cache[cache_key]
        if debug:
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
    for f in fs:
        ans = f(frag)
        if ans is not None:
            assert isinstance(ans, bool)
            if ans:
                return True

    return False


def run_phases(frag: Fragment, *fs: Callable[[Fragment], bool | None], loop: bool = True) -> bool:
    if not loop:
        return _run_phases_once(frag, *fs)

    res = False
    while _run_phases_once(frag, *fs):
        res = True
    return res
