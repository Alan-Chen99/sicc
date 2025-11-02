from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Concatenate
from typing import Final
from typing import Self
from typing import TypedDict
from typing import Unpack
from typing import overload

from . import _functions as _f
from ._api import TreeSpec
from ._api import Variable
from ._api import _get_type
from ._api import copy_tree
from ._api import jump
from ._api import read_uservalue
from ._control_flow import block
from ._core import Label
from ._core import Var
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import debug_info
from ._diagnostic import describe_fn
from ._state import State
from ._tracing import _CUR_SCOPE
from ._tracing import RawSubr
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import trace_to_raw_subr
from ._utils import Cell
from ._utils import get_id


@dataclass
class TracedSubr[F = Any]:
    subr: RawSubr
    arg_tree: TreeSpec
    ret_tree: TreeSpec

    @property
    def call[**P, R](self: TracedSubr[Callable[P, R]]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            vals = self.arg_tree.flatten_up_to((args, kwargs))
            vals_ = tuple(read_uservalue(x) for x in vals)
            out_vars = self.subr.call(*vals_)

            out_vars_ = [Variable._from_val_ro(x) for x in out_vars]
            return self.ret_tree.unflatten(out_vars_)

        return inner


_RETURN_HOOK: Cell[Callable[[Any], None]] = Cell()


def return_(val: Any = None) -> None:
    return _RETURN_HOOK(val)


def trace_to_subr[**P, R](
    fn: Callable[P, R | None], *args: P.args, **kwargs: P.kwargs
) -> TracedSubr[Callable[P, R]]:
    arg_vars, arg_tree = TreeSpec.flatten((args, kwargs))
    arg_types = tuple(_get_type(x) for x in arg_vars)

    out_tree: Cell[TreeSpec[R]] = Cell()

    @functools.wraps(fn)
    def inner(*args: Var) -> tuple[Var, ...]:
        ar, kw = arg_tree.unflatten(Variable._from_val_ro(x) for x in args)

        id = get_id()

        subr_scope = _CUR_SCOPE.value
        exit_paths: list[tuple[Label, R, DebugInfo]] = []

        def ret_hook(val: Any) -> None:
            with _CUR_SCOPE.bind(subr_scope):
                exit_label = mk_internal_label(f"trace_to_subr_ret({len(exit_paths)})", id)
                # FIXME: this is a out of scope use;
                # we dont currenty error for that but will probably in future
                exit_paths.append((exit_label, copy_tree(val), debug_info()))
                jump(exit_label)

        with _RETURN_HOOK.bind(ret_hook):
            ans = fn(*ar, **kw)

            with add_debug_info(DebugInfo(describe=f"return val from end of {describe_fn(fn)}")):
                return_(ans)
                _f.unreachable_checked()

        ret_tree = TreeSpec.promote_types_many(
            *(TreeSpec.flatten(x)[1] for _, x, _ in exit_paths),
        )
        out_tree.value = ret_tree
        ret_state = State(_tree=ret_tree)

        actual_exit = mk_internal_label(f"trace_to_subr_exit", id)

        for exit_label, exit_val, debug in exit_paths:
            with add_debug_info(debug):
                label(exit_label)
                ret_state.write(exit_val)
                jump(actual_exit)

        label(actual_exit)

        assert ret_state._vars is not None
        return tuple(x.read() for x in ret_state._vars)

    ans = trace_to_raw_subr(arg_types, inner)

    return TracedSubr(ans, arg_tree, out_tree.value)


class Subr[F]:
    fn: Final[F]

    def __init__(self, fn: F) -> None:
        self.fn = fn
        self._subr: TracedSubr | None = None

    def __repr__(self):
        return repr(self.fn)

    @overload
    def __get__(self, obj: None, objtype: type, /) -> Self: ...
    @overload
    def __get__[T, **P, R](
        self: Subr[Callable[Concatenate[T, P], R]], obj: T, objtype: Any, /
    ) -> Callable[P, R]: ...

    def __get__(self, obj: Any, objtype: Any, /):
        if obj is None:
            return self

        def inner(*args: Any, **kwargs: Any) -> Any:
            return self(
                obj, *args, **kwargs
            )  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

        return inner

    def __call__[**P, R](self: Subr[Callable[P, R]], *args: P.args, **kwargs: P.kwargs) -> R:
        bound = inspect.signature(self.fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if self._subr is None:
            self._subr = trace_to_subr(
                self.fn, **bound.arguments
            )  # pyright: ignore[reportCallIssue]
        return self._subr.call(**bound.arguments)


@dataclass
class SubrFactory:
    inline_always: bool = False

    def __call__[F](self, fn: F) -> Subr[F]:
        if self.inline_always:
            raise NotImplementedError()
        return Subr(fn)


class SubrOpts(TypedDict, total=False):
    inline_always: bool


@overload
def subr(**kwargs: Unpack[SubrOpts]) -> SubrFactory: ...
@overload
def subr[F](func: F, /, **kwargs: Unpack[SubrOpts]) -> Subr[F]: ...


def subr[F](func: F | None = None, /, **kwargs: Unpack[SubrOpts]) -> SubrFactory | Subr[F]:
    config = SubrFactory(**kwargs)
    if func is None:
        return config
    return config(func)


def inline_subr[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with block() as b:

            def ret_hook(val: Any) -> None:
                b.break_(val)

            with _RETURN_HOOK.bind(ret_hook):
                ans = fn(*args, **kwargs)
                return_(ans)
                _f.unreachable_checked()

        return b.value

    return inner
