from __future__ import annotations

# pyright: reportUnusedFunction=false, reportUnusedImport=false

import abc
import ast
import inspect
import types
from abc import abstractmethod
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Generator
from typing import Never
from typing import cast
from typing import overload

import optree
from rich.console import RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from ._api import Bool
from ._api import Variable
from ._api import not_
from ._api import read_uservalue
from ._api import undef
from ._control_flow import break_
from ._control_flow import cond
from ._control_flow import continue_
from ._control_flow import else_
from ._control_flow import if_
from ._control_flow import while_
from ._coroutine import static_coroutine
from ._diagnostic import register_exclusion
from ._iterator import runtime_loop
from ._iterator import shared_generator
from ._state import State
from ._subr import return_
from ._subr import subr
from ._tracing import mk_internal_label
from ._transpile_utils import FunctionDefNode
from ._transpile_utils import resolve_function_scope
from ._transpile_utils import resolve_scope_shadowed_names
from ._transpile_utils import transpile_runtime
from ._tree_utils import PyTree
from ._tree_utils import TreeSpec
from ._tree_utils import ValSpec
from ._tree_utils import pytree
from ._tree_utils import tree_flatten
from ._utils import CallDescMixin
from ._utils import Cell
from ._utils import Empty
from ._utils import empty
from ._utils import get_id

register_exclusion(__file__)

# any identifier or attribute starting with _PREFIX raises _TranspileErrorWrapped
_PREFIX = "_sicc"
# import sicc as _sicc at the top
_SICC_IMPORT = _PREFIX


class TranspileError(Exception):
    pass


class TranspileNotImplementedError(TranspileError, NotImplementedError):
    pass


class TranspileReservedPrefixError(TranspileError, TypeError):
    pass


@dataclass
class _Place:
    """
    A mutable "place" used by @transpile to model locals and lvalues.

    Places store a `State` whose leaves are addressed via `Variable` objects.
    Writing updates existing storage (never rebinding leaves), which ensures:
    - args can be mutable while preserving identity of the underlying storage
    - locals never alias each other (assignment copies values into local storage)
    """

    state: State[Any] = field(default_factory=State)

    def _err_if_unbound(self):
        if not self.state._did_init():
            raise UnboundLocalError("local variable referenced before assignment")

    @property
    def value(self) -> Any:
        self._err_if_unbound()
        return self.state.ref_mut()

    @value.setter
    def value(self, val: Any) -> None:
        self.state.write(val)

    @value.deleter
    def value(self) -> None:
        self._err_if_unbound()

        leaf_vars, _tree = TreeSpec.flatten(self.state.ref_mut())
        for v in leaf_vars:
            assert isinstance(v, Variable)
            v.value = undef(v._get_type())


@dataclass
class _AttrProxy:
    _obj: Any
    _name: str

    @property
    def value(self) -> Any:
        return getattr(self._obj, self._name)

    @value.setter
    def value(self, val: Any) -> None:
        if (setfn := getattr(self._obj, "__transpile_setattr__", None)) is not None:
            setfn(self._name, val)
            return
        _write_place(self.value, val)


def _attr(obj: Any, name: str) -> _AttrProxy:
    return _AttrProxy(obj, name)


@dataclass
class _ItemProxy:
    _obj: Any
    _key: Any

    @property
    def value(self) -> Any:
        return self._obj[self._key]

    @value.setter
    def value(self, val: Any) -> None:
        if (setitem := getattr(self._obj, "__transpile_setitem__", None)) is not None:
            setitem(self._key, val)
            return
        _write_place(self.value, val)


@dataclass
class _SubsProxy:
    _obj: Any

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __getitem__(self, key: Any) -> _ItemProxy:
        return _ItemProxy(self._obj, key)


def _subs(obj: Any) -> _SubsProxy:
    return _SubsProxy(obj)


def _init_place(init_value: Any | Empty = empty) -> _Place:
    if init_value is not empty:
        return _Place(State.from_ref_mut(init_value))
    return _Place()


def _write_place[T](dst: T, src: T) -> None:
    dst_leaves, dst_tree = TreeSpec.flatten(dst)
    src_leaves = dst_tree.flatten_up_to(src, check_mut=False)

    if any(not x.mutable for x in dst_tree.leaves):
        raise TypeError(f"cannot write to {dst_tree} because it is not mutable")

    for v, x in zip(dst_leaves, src_leaves):
        assert isinstance(v, Variable)
        v.value = x


def _call(fn: Any, /, *args: Any, **kwargs: Any) -> Any:
    """
    Wrap calls inside @transpile.

    - Verifies that the pytree structure and leaf identities of inputs are unchanged.
    - Executes the call.
    """

    arg_objs, arg_tree = tree_flatten(cast(PyTree[Any], (fn, args, kwargs)))
    res = fn(*args, **kwargs)
    arg_objs_post, arg_tree_post = tree_flatten(cast(PyTree[Any], (fn, args, kwargs)))

    if arg_tree != arg_tree_post or any(x is not y for x, y in zip(arg_objs, arg_objs_post)):
        prev = arg_tree.unflatten(arg_objs)
        post = arg_tree_post.unflatten(arg_objs_post)
        raise TypeError(f"call to {fn} modified argument: {prev} -> {post}")

    return res


def _and(x: Bool, rhs: Callable[[], Bool]) -> Bool:
    """Short-circuiting boolean AND that matches Python's evaluation rules."""
    if isinstance(x, bool):
        return rhs() if x else x
    return cond(x, rhs, x)


def _or(x: Bool, rhs: Callable[[], Bool]) -> Bool:
    """Short-circuiting boolean OR that matches Python's evaluation rules."""
    if isinstance(x, bool):
        return x if x else rhs()
    return cond(x, x, rhs)


def _not(x: Bool) -> Bool:
    # not_ is not function so does not work with _embed_object
    return not_(x)


def _embed_object(obj: Any) -> ast.expr:
    """Return an AST expression that resolves to `obj` via the `sicc` package.

    The returned expression uses the `_sicc` alias (injected into transpiled functions) and a
    dotted attribute path derived from `obj.__module__` and `obj.__qualname__`.
    """
    path: str = obj.__module__ + "." + obj.__qualname__
    parts = path.split(".")

    assert parts[0] == "sicc"

    expr = ast.Name(id=_PREFIX, ctx=ast.Load())
    for attr in parts[1:]:
        expr = ast.Attribute(value=expr, attr=attr, ctx=ast.Load())

    return expr


def _embed_call[**P](fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> ast.Call:
    def _as_ast(val: Any) -> ast.expr:
        if isinstance(val, ast.AST):
            assert isinstance(val, ast.expr)
            return val
        return ast.Constant(value=val)

    return ast.Call(
        func=_embed_object(fn),
        args=[_as_ast(v) for v in args],
        keywords=[ast.keyword(k, _as_ast(v)) for k, v in kwargs.items()],
    )


@dataclass(frozen=True)
class _TranspileErrorWrapped(Exception):
    node: ast.AST
    err_type: type[TranspileError]
    msg: str

    def __str__(self) -> str:
        return self.msg


class _TranspileBodyTransformer(ast.NodeTransformer):
    def __init__(self, locals_: set[str]) -> None:
        self._locals: Cell[set[str]] = Cell(locals_)

    def _visit_sametype[N: ast.AST](self, node: N) -> N:
        typ = type(node)
        ans = self.visit(node)
        assert isinstance(ans, typ)
        return ans

    def _generic_visit_sametype[N: ast.AST](self, node: N) -> N:
        typ = type(node)
        ans = self.generic_visit(node)
        assert isinstance(ans, typ)
        return ans

    @contextmanager
    def _exclude_locals(self, names: set[str]) -> Iterator[None]:
        if not names:
            yield None
            return
        with self._locals.bind(self._locals.value - names):
            yield None

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    #####

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "nesting")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "nesting")

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "class def")

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        with self._exclude_locals(resolve_scope_shadowed_names(node)):
            return self._generic_visit_sametype(node)

    #####

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self._locals.value:
            return node

        assert isinstance(node.ctx, (ast.Load, ast.Store, ast.Del))
        expr = ast.Attribute(value=ast.Name(id=node.id, ctx=ast.Load()), attr="value", ctx=node.ctx)
        return ast.copy_location(expr, node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self._generic_visit_sametype(node)
        if not isinstance(node.ctx, (ast.Store, ast.Del)):
            return node

        place = _embed_call(_attr, node.value, cast(Any, ast.Constant(value=node.attr)))
        expr = ast.Attribute(value=place, attr="value", ctx=node.ctx)
        return ast.copy_location(expr, node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self._generic_visit_sametype(node)
        if not isinstance(node.ctx, (ast.Store, ast.Del)):
            return node

        place = ast.Subscript(
            value=_embed_call(_subs, node.value), slice=node.slice, ctx=ast.Load()
        )
        expr = ast.Attribute(value=place, attr="value", ctx=node.ctx)
        return ast.copy_location(expr, node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self._generic_visit_sametype(node)

        expr = ast.Call(
            func=_embed_object(_call), args=[node.func, *node.args], keywords=node.keywords
        )
        return ast.copy_location(expr, node)


def _assert_no_reserved_prefix(fn_node: FunctionDefNode) -> None:
    def check(node: ast.AST, name: str | None, desc: str) -> None:
        if name is not None and name.startswith(_PREFIX):
            raise _TranspileErrorWrapped(
                node,
                TranspileReservedPrefixError,
                f"@transpile reserves prefix {_PREFIX!r}; found {desc} {name!r}",
            )

    for node in ast.walk(fn_node):
        match node:
            case ast.Name(id=name):
                check(node, name, "name")
            case ast.arg(arg=name):
                check(node, name, "arg")
            case ast.Attribute(attr=name):
                check(node, name, "attribute")
            case ast.keyword(arg=name):
                check(node, name, "keyword")
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                check(node, name, "function")
            case ast.ClassDef(name=name):
                check(node, name, "class")
            case ast.alias(name=name, asname=asname):
                check(node, name, "import")
                check(node, asname, "import alias")
            case ast.ExceptHandler(name=name):
                check(node, name, "except name")
            case ast.Global(names=names) | ast.Nonlocal(names=names):
                for name in names:
                    check(node, name, "name")
            case ast.MatchAs(name=name) | ast.MatchStar(name=name):
                check(node, name, "match name")
            case _:
                pass


def _transpile_ast(fn_node: FunctionDefNode) -> FunctionDefNode:
    try:
        return _transpile_ast_impl(fn_node)
    except _TranspileErrorWrapped as e:
        # Avoid import-time failures while still getting a runtime traceback pointing at the
        # unsupported syntax / reserved name in the original source.
        fn_node.body = [
            ast.Import(names=[ast.alias(name="sicc", asname=_SICC_IMPORT)]),
            ast.copy_location(ast.Raise(exc=_embed_call(e.err_type, e.msg)), e.node),
        ]
        return fn_node


def _transpile_ast_impl(fn_node: FunctionDefNode) -> FunctionDefNode:
    _assert_no_reserved_prefix(fn_node)
    locals_, params = resolve_function_scope(fn_node)
    places = locals_ | params

    transformer = _TranspileBodyTransformer(places)
    fn_node.body = [transformer._visit_sametype(x) for x in fn_node.body]

    init: list[ast.stmt] = [
        ast.Import(names=[ast.alias(name="sicc", asname=_SICC_IMPORT)]),
        *[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=_embed_call(_init_place, ast.Name(id=name, ctx=ast.Load())),
            )
            for name in sorted(params)
        ],
        *[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=_embed_call(_init_place),
            )
            for name in sorted(locals_)
        ],
    ]

    insert_at = 0
    if (
        fn_node.body
        and isinstance(fn_node.body[0], ast.Expr)
        and isinstance(fn_node.body[0].value, ast.Constant)
        and isinstance(fn_node.body[0].value.value, str)
    ):
        insert_at = 1

    fn_node.body[insert_at:insert_at] = init
    return fn_node


@dataclass
class Transpiled[**P, R](CallDescMixin):
    orig: Callable[P, R]
    transpiled: Callable[P, R]
    orig_str: str
    transpiled_str: str

    def __post_init__(self) -> None:
        self.__name__ = self.orig.__name__
        self.__qualname__ = self.orig.__qualname__
        self.__module__ = self.orig.__module__
        self.__wrapped__ = self.orig

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.transpiled(*args, **kwargs)

    def __str__(self) -> str:
        return self.transpiled_str

    def __repr__(self) -> str:
        return repr(self.orig)

    def pretty(self) -> RenderableType:
        return Panel(
            Syntax(self.transpiled_str, "python", line_numbers=True, theme="ansi_dark"),
            title=Text(f"Transpiled: {self.orig.__qualname__}", "ic10.title"),
            title_align="left",
        )


@overload
def transpile[**P, R](fn: Callable[P, Awaitable[R]]) -> Transpiled[P, Awaitable[R]]: ...
@overload
def transpile[**P, Y, R](
    fn: Callable[P, Generator[Y, None, R]],
) -> Transpiled[P, Generator[Y, None, R]]: ...
@overload
def transpile[**P, R](fn: Callable[P, R]) -> Transpiled[P, R]: ...


def transpile[**P, R](fn: Callable[P, R]) -> Transpiled[P, R]:
    """
    transform a function to be one suitable for tracing. designed to be transparent.

    **printed output**
    `rich.print(transpiled_fn.pretty())` prints the transpiled source. this is the code that will be executed
    (conceptually as if it replaced the original function), but an internal factory wrapper is used
    to capture closures correctly.

    **local variables**
    locals and parameters are rewritten to `_Place` instances.
    reads/writes go through the `.value` property:
    - `a` -> `a.value`
    - `a.b` -> `a.value.b`
    - `a[b]` -> `a.value[b]`

    attribute/item stores are wrapped to preserve structural assignment semantics:
    - `a.b = x` -> `_attr(a.value, "b").value = x`
    - `a[b] = x` -> `_subs(a.value)[b].value = x`

    **non-local lookups**
    variables that are not locals to `fn` under @transpile are kept as is. this includes:
    (1) vars that refers to outside `fn`
    (2) vars in inner scope like args or local vars of inner def, including those shadowing locals of `fn`
    `nonlocal` stays; inner references are recognized as refereing the outer one and therefore rewritten.

    **return wrapping**
    `def` wrapped with `subr`
    `async def` wrapped with `static_coroutine`
    generator functions wrapped with `shared_generator`

    **control flow**
    `if x: ... else: ...` -> `with if_(x): ... with else_(): ...`
    `x if y else z` -> `cond(y, lambda: x, lambda: z)`
    `while x` -> `with while_(lambda: x)`
    `for x in y` -> `for x in patch_for_loop(y)`
    `break` -> `break_`
    `continue` -> `continue_`
    `return` -> `return_`
    `del x` -> `del x.value` (implemented as set to undef)

    **expression**
    `not x` -> `_not(x)`
    `x and y` -> `_and(x, lambda: y)`
    `x or y` -> `_or(x, lambda: y)`

    **planned (raises TranspileNotImplementedError currently)**
    `match` -> as expected
    `:=` -> as expected
    `for/else` -> as expected
    `while/else` -> as expected
    `global` -> `_Place` object stored in global scope

    **all other contstructs**
    kept as is
    comprehensions runs at trace time

    **note on unsupported syntax**
    unsupported constructs are rewritten into a runtime `raise` so that
    transpilation can happen at import-time without crashing, while still producing a useful
    backtrace pointing at the original source location when executed.
    """

    transpiled_fn, source_original, source_wrapper = transpile_runtime(fn, transform=_transpile_ast)

    wrap_fn: Any

    if inspect.iscoroutinefunction(fn):
        wrap_fn = static_coroutine
    elif inspect.isgeneratorfunction(fn):
        wrap_fn = shared_generator
    else:
        wrap_fn = subr

    ans = Transpiled[P, R](
        orig=fn,  # pyright: ignore[reportArgumentType]
        transpiled=wrap_fn(transpiled_fn),
        orig_str=source_original,
        transpiled_str=source_wrapper,
    )

    # transpile happen at import time where things are not properly setup
    # so we cant do this rn
    # if verbose.value >= 1:
    #     print(ans.pretty())

    return ans
