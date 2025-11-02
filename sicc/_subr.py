from __future__ import annotations

import functools
from dataclasses import dataclass
from types import GeneratorType
from typing import Any
from typing import Callable
from typing import Concatenate
from typing import Generator
from typing import Generic
from typing import Self
from typing import TypedDict
from typing import TypeVar
from typing import Unpack
from typing import cast
from typing import overload

from ._api import Variable
from ._api import read_uservalue
from ._control_flow import BlockRef
from ._control_flow import block
from ._control_flow import clear_control_flow_hooks
from ._core import Value
from ._core import Var
from ._diagnostic import register_exclusion
from ._instructions import unreachable_checked
from ._tracing import RawSubr
from ._tracing import trace_to_raw_subr
from ._tree_utils import TreeSpec
from ._utils import Cell
from ._utils import cast_unchecked_val
from ._utils import normalize_function_args

register_exclusion(__file__)


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


R_co = TypeVar("R_co", covariant=True, default=Any)


@dataclass
class Return(Generic[R_co]):
    value: R_co

    def __init__(self, val: R_co = None, /):
        self.value = val


type FunctionRet[R] = Generator[Return[R], None, None] | R


def trace_to_subr[**P, R](
    fn: Callable[P, FunctionRet[R]],
    arg_tree_: TreeSpec | None,
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> TracedSubr[Callable[P, R]]:
    _arg_vars, arg_tree = TreeSpec.flatten((args, kwargs))
    if arg_tree_ is not None:
        arg_tree.can_cast_implicit_many_or_err(arg_tree_)
        arg_tree = cast_unchecked_val(arg_tree)(arg_tree_)

    out_tree: Cell[TreeSpec[R]] = Cell()

    @functools.wraps(fn)
    def inner(*args: Var) -> tuple[Value, ...]:
        ar, kw = arg_tree.unflatten(Variable._from_val_ro(x) for x in args)

        ans = inline_subr(fn)(*ar, **kw)

        vals, tree = TreeSpec.flatten(ans)
        out_tree.value = tree
        return tuple(read_uservalue(x) for x in vals)

    ans = trace_to_raw_subr(arg_tree.types, inner)

    return TracedSubr(ans, arg_tree, out_tree.value)


@dataclass
class ArgSpec:
    _spec: TreeSpec[tuple[tuple[Any, ...], dict[str, Any]]]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._spec = TreeSpec.from_schema((args, kwargs))


F_co = TypeVar("F_co", covariant=True)


@dataclass
class Subr(Generic[F_co]):
    fn: F_co
    arg_types: ArgSpec | None = None

    _subr: TracedSubr[F_co] | None = None

    def __repr__(self):
        return repr(self.fn)

    def __post_init__(self):
        self.__qualname__ = self.fn.__qualname__

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
        args, kwargs = normalize_function_args(self.fn, args, kwargs)

        if self._subr is None:
            if self.arg_types is not None:
                spec_ar, spec_kw = self.arg_types._spec.as_schema()
                arg_types = TreeSpec.from_schema(normalize_function_args(self.fn, spec_ar, spec_kw))
            else:
                arg_types = None

            self._subr = trace_to_subr(self.fn, arg_types, *args, **kwargs)

        return self._subr.call(*args, **kwargs)


class _Invalid:
    pass


@dataclass
class SubrFactory:
    inline_always: bool = False
    arg_types: ArgSpec | None = None

    @overload
    def __call__[**P, R](
        self, func: Callable[P, Generator[Return[R]]], /
    ) -> Subr[Callable[P, R]]: ...
    @overload
    def __call__[**P, R](  # pyright: ignore[reportOverlappingOverload]
        self, func: Callable[P, Generator[Any, Any, Any]], /
    ) -> Subr[_Invalid]: ...
    @overload
    def __call__[**P, R](self, func: Callable[P, FunctionRet[R]]) -> Subr[Callable[P, R]]: ...

    def __call__[**P, R](  # pyright: ignore[reportInconsistentOverload]
        self, func: Callable[P, FunctionRet[R]], /
    ) -> Subr[Callable[P, R]]:
        if self.inline_always:
            raise NotImplementedError()
        return Subr(cast(Callable[P, R], func), self.arg_types)


class SubrOpts(TypedDict, total=False):
    inline_always: bool
    arg_types: ArgSpec | None


@overload
def subr(**kwargs: Unpack[SubrOpts]) -> SubrFactory: ...
@overload
def subr[**P, R](
    func: Callable[P, Generator[Return[R]]], /, **kwargs: Unpack[SubrOpts]
) -> Subr[Callable[P, R]]: ...
@overload
def subr[**P, R](  # pyright: ignore[reportOverlappingOverload]
    func: Callable[P, Generator[Any, Any, Any]], /, **kwargs: Unpack[SubrOpts]
) -> Subr[_Invalid]: ...
@overload
def subr[**P, R](
    func: Callable[P, FunctionRet[R]], /, **kwargs: Unpack[SubrOpts]
) -> Subr[Callable[P, R]]: ...


def subr[**P, R](  # pyright: ignore[reportInconsistentOverload]
    func: Callable[P, FunctionRet[R]] | None = None, /, **kwargs: Unpack[SubrOpts]
):
    config = SubrFactory(**kwargs)
    if func is None:
        return config
    return config(func)


_RETURN_HOOK: Cell[Callable[[Any], None]] = Cell()


def return_(val: Any = None) -> None:
    return _RETURN_HOOK(val)


def inline_subr[**P, R](fn: Callable[P, FunctionRet[R]]) -> Callable[P, R]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with (
            block(BlockRef[R]) as b,
            clear_control_flow_hooks(),
            _RETURN_HOOK.bind(b.break_),
        ):
            ans = fn(*args, **kwargs)

            if isinstance(ans, GeneratorType):
                ans = cast(Generator[Return[R]], ans)
                try:
                    while True:
                        item = next(ans)
                        if not isinstance(
                            item, Return
                        ):  # pyright: ignore[reportUnnecessaryIsInstance]
                            raise TypeError()
                        b.break_(item.value)
                except StopIteration as e:
                    if e.value is not None:
                        raise TypeError("Generator-based subr should not also return") from None
                unreachable_checked(
                    f"Generator-based subr {ans.__qualname__} is required to return at the end"
                )
            else:
                ans = cast(R, ans)
                b.break_(ans)

        return b.value

    return inner
