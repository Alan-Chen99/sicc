from __future__ import annotations

import ast
import copy
import functools
import inspect
import types
from contextlib import AbstractContextManager
from contextlib import contextmanager
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import TypeVar
from typing import cast
from typing import get_origin
from typing import overload

from rich.console import RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from ._api import Bool
from ._api import Variable
from ._api import not_
from ._api import undef
from ._control_flow import BREAK_TO
from ._control_flow import CONTINUE_TO
from ._control_flow import ELSE_HOOK
from ._control_flow import break_
from ._control_flow import cond
from ._control_flow import continue_
from ._control_flow import else_
from ._control_flow import if_
from ._control_flow import while_
from ._core import Label
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._iterator import shared_generator
from ._state import State
from ._subr import return_
from ._subr import subr
from ._tracing import label
from ._tracing import mk_internal_label
from ._transpile_utils import EvalExpr
from ._transpile_utils import FunctionDefNode
from ._transpile_utils import resolve_function_scope
from ._transpile_utils import resolve_scope_shadowed_names
from ._transpile_utils import transpile_runtime
from ._tree_utils import PyTree
from ._tree_utils import TreeSpec
from ._tree_utils import tree_flatten
from ._tree_utils import tree_write
from ._utils import CallDescMixin
from ._utils import Cell
from ._utils import Empty
from ._utils import allow_curry
from ._utils import empty
from ._utils import get_id
from ._utils import identity

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


class _StaticMarker:
    def __repr__(self) -> str:
        return "Static"


_STATIC = _StaticMarker()

T = TypeVar("T")

Static = Annotated[T, _STATIC]
"""
Annotation marking a local or parameter of a @transpile function as static: it is a plain
trace-time python variable, left untouched by @transpile. See `transpile`.
"""


def _is_static_annotation(val: Any) -> bool:
    return get_origin(val) is Annotated and any(m is _STATIC for m in val.__metadata__)


@dataclass
class _Place:
    """
    A mutable "place" used by @transpile to model locals and lvalues.

    Places store a `State` whose leaves are addressed via `Variable` objects.
    Writing updates existing storage (never rebinding leaves), which ensures:
    - args can be mutable while preserving identity of the underlying storage
    - locals never alias each other (assignment copies values into local storage)
    """

    state: State[Any]

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
        tree_write(self.value, val)


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
        tree_write(self.value, val)


@dataclass
class _SubsProxy:
    _obj: Any

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __getitem__(self, key: Any) -> _ItemProxy:
        return _ItemProxy(self._obj, key)


def _subs(obj: Any) -> _SubsProxy:
    return _SubsProxy(obj)


def _init_place(name: str, init_value: Any | Empty = empty) -> _Place:
    with add_debug_info(DebugInfo(describe=f"transpiled variable [ic10.label]{name}[/ic10.label]")):
        if init_value is not empty:
            return _Place(State._from_ref_mut(init_value))
        return _Place(State())


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


def _patch_for_loop[E](it: Iterable[E], *, break_to: Label) -> Iterator[E]:
    """
    make continue, break, else work
    note that only transpile support for..else; tracing does not.
    """
    id = get_id()

    for idx, item in enumerate(it):
        with track_caller():
            for_cont = mk_internal_label(f"for_cont({idx})", id)

        with (
            CONTINUE_TO.bind(for_cont),
            BREAK_TO.bind(break_to),
            ELSE_HOOK.bind_clear(),
        ):
            yield item

        with track_caller():
            label(for_cont)


def _and(x: Bool, rhs: Callable[[], Bool]) -> Bool:
    """Short-circuiting boolean AND that matches Python's evaluation rules."""
    if isinstance(x, bool):
        return x and rhs()

    x = Variable(x)
    ans = Variable(False)
    with if_(x):
        ans.value = rhs()
    return ans.value


def _or(x: Bool, rhs: Callable[[], Bool]) -> Bool:
    """Short-circuiting boolean OR that matches Python's evaluation rules."""
    if isinstance(x, bool):
        return x or rhs()

    x = Variable(x)
    ans = Variable(True)
    with if_(not_(x)):
        ans.value = rhs()
    return ans.value


def _ifexp[T](test: Bool, body: Callable[[], T], orelse: Callable[[], T]) -> T:
    """`body() if test else orelse()`; a python bool `test` picks the branch at trace time."""
    if isinstance(test, bool):
        return body() if test else orelse()
    return cond(test, body, orelse)


def _if_body(test: Bool) -> AbstractContextManager[None]:
    """`if_(test)`; a python bool `test` selects a branch at trace time instead (see visit_If)."""
    if isinstance(test, bool):
        return nullcontext()
    return if_(test)


def _if_else(test: Bool) -> AbstractContextManager[None]:
    """`else_()` for `_if_body(test)`."""
    if isinstance(test, bool):
        return nullcontext()
    return else_()


def _not(x: Bool) -> Bool:
    if isinstance(x, bool):
        return not x

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

    expr = ast.Name(id=_SICC_IMPORT, ctx=ast.Load())
    for attr in parts[1:]:
        expr = ast.Attribute(value=expr, attr=attr, ctx=ast.Load())

    return expr


@overload
def _embed_call[**P](fn: Callable[P, Any], *args: P.args, **kwargs: P.kwargs) -> ast.Call: ...
@overload
def _embed_call[**P](fn: Callable[P, Any], *args: ast.expr) -> ast.Call: ...
@overload
def _embed_call[**P](fn: Callable[P, Any], *args: ast.expr, **kwargs: ast.expr) -> ast.Call: ...


def _embed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> ast.Call:
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


@dataclass
class _TranspileErrorWrapped(Exception):
    node: ast.AST
    err_type: type[TranspileError]
    msg: str

    def __str__(self) -> str:
        return self.msg


@dataclass(frozen=True)
class _NestedDef:
    """A `def` nested in a @transpile function, kept so `@transpile` can be applied to it."""

    node: FunctionDefNode  # original AST, absolute source positions
    outer_places: frozenset[str]  # non-static locals of enclosing @transpile scopes it can see


_NESTED_DEFS: list[_NestedDef] = []
_NESTED_DEF_BY_CODE: dict[types.CodeType, _NestedDef] = {}


def _mark_nested_def[F](idx: int) -> Callable[[F], F]:
    """Decorator injected on nested defs; maps the compiled code object to its `_NestedDef`."""

    def mark(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        _NESTED_DEF_BY_CODE[fn.__code__] = _NESTED_DEFS[idx]
        return fn

    return mark


def _iter_scope(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Yield `nodes` and their descendants within the same scope.

    Bodies of nested defs, classes and lambdas are their own scope and are not entered.
    """
    for node in nodes:
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from _iter_scope(ast.iter_child_nodes(node))


def _import_binding(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".")[0]


def _resolve_statics(
    fn_node: FunctionDefNode,
    scope_names: set[str],
    eval_expr: EvalExpr,
    param_annotations: dict[str, Any],
) -> set[str]:
    """Return the names in `scope_names` that are static in `fn_node`.

    A name is static if it has a `Static` annotation, or is bound by `def`, `class` or `import`
    (a `_Place` cannot hold functions or modules).

    `param_annotations`: `__annotations__` of the function; non-string values were evaluated
    by python in the right scope (e.g. class bodies for methods) and are used as is.
    """
    # name -> [(node, is_static)]
    evidence: dict[str, list[tuple[ast.AST, bool]]] = {}

    def annotated(node: ast.AST, name: str, ann: ast.expr) -> None:
        if isinstance(node, ast.arg) and name in param_annotations:
            val = param_annotations[name]
            if not isinstance(val, str):
                evidence.setdefault(name, []).append((node, _is_static_annotation(val)))
                return
        try:
            val = eval_expr(ann)
        except Exception as e:
            raise _TranspileErrorWrapped(
                node,
                TranspileError,
                f"evaluating annotation {ast.unparse(ann)!r} of {name!r} failed: {e!r}",
            ) from e
        evidence.setdefault(name, []).append((node, _is_static_annotation(val)))

    args = fn_node.args
    for arg in [*args.posonlyargs, *args.args, args.vararg, *args.kwonlyargs, args.kwarg]:
        if arg is not None and arg.annotation is not None:
            annotated(arg, arg.arg, arg.annotation)

    for node in _iter_scope(fn_node.body):
        match node:
            case ast.AnnAssign(target=ast.Name(id=name), annotation=ann):
                annotated(node, name, ann)
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(
                name=name
            ):
                evidence.setdefault(name, []).append((node, True))
            case ast.alias():
                evidence.setdefault(_import_binding(node), []).append((node, True))
            case _:
                pass

    statics: set[str] = set()
    for name, items in evidence.items():
        if name not in scope_names:
            continue
        if not any(is_static for _, is_static in items):
            continue
        for node, is_static in items:
            if not is_static:
                raise _TranspileErrorWrapped(
                    node,
                    TranspileError,
                    f"{name!r} is both static (Static annotation, def, class or import) "
                    "and annotated as non-static",
                )
        statics.add(name)
    return statics


def _check_static_stores(fn_node: FunctionDefNode, places: set[str]) -> None:
    """Reject assignments to non-place names under runtime control flow.

    `if_` traces both branches and `while_` traces its body once, so an assignment to a
    plain python name (Static, def, import, nonlocal or global) there runs once at trace time,
    regardless of the runtime condition.
    """

    def check(node: ast.AST, name: str, conditional: bool) -> None:
        if conditional and name not in places:
            raise _TranspileErrorWrapped(
                node,
                TranspileError,
                f"assignment to {name!r} under a runtime `if`/`while`/`and`/`or`/ternary: "
                f"{name!r} is not a runtime local (it is Static, def, class, import, "
                "nonlocal or global), so the assignment runs once at trace time regardless of the condition",
            )

    def walk(node: ast.AST, conditional: bool) -> None:
        match node:
            case ast.If(test=test, body=body, orelse=orelse):
                walk(test, conditional)
                for stmt in [*body, *orelse]:
                    walk(stmt, True)
            case ast.While(test=test, body=body, orelse=orelse):
                for sub in [test, *body, *orelse]:
                    walk(sub, True)
            case ast.IfExp(test=test, body=body, orelse=orelse):
                walk(test, conditional)
                walk(body, True)
                walk(orelse, True)
            case ast.BoolOp(values=[first, *rest]):
                walk(first, conditional)
                for sub in rest:
                    walk(sub, True)
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                check(node, node.name, conditional)
                for sub in node.decorator_list:
                    walk(sub, conditional)
            case ast.Lambda() | ast.ListComp() | ast.SetComp() | ast.GeneratorExp() | ast.DictComp():
                pass  # own scope
            case ast.AnnAssign(value=None):
                pass
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                check(node, name, conditional)
            case ast.alias():
                check(node, _import_binding(node), conditional)
            case _:
                if isinstance(node, ast.ExceptHandler) and node.name is not None:
                    check(node, node.name, conditional)
                for sub in ast.iter_child_nodes(node):
                    walk(sub, conditional)

    for stmt in fn_node.body:
        walk(stmt, False)


class _TranspileBodyTransformer(ast.NodeTransformer):
    def __init__(self, places: set[str]) -> None:
        # non-static locals visible in the current scope; rewritten to `_Place` accesses
        self._places: Cell[set[str]] = Cell(places)
        # inside a nested def: trace-time python where only place accesses are rewritten
        self._helper: Cell[bool] = Cell(False)
        # whether the value of the statement whose targets are being visited references a place
        self._stmt_refs_places: Cell[bool] = Cell(False)
        # count of rewritten place references; a visit that leaves it unchanged touched no place
        self._place_refs: int = 0
        self._tmp_id: int = 0

    def _tmp(self, name: str) -> str:
        """Create a unique internal name guaranteed not to collide with user code."""
        ans = f"{_PREFIX}_{name}_{self._tmp_id}"
        self._tmp_id += 1
        return ans

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

    def _visit_expr(self, node: ast.expr) -> ast.expr:
        ans = self.visit(node)
        assert isinstance(ans, ast.expr)
        return ans

    def _visit_value(self, node: ast.expr) -> tuple[ast.expr, bool]:
        """Visit `node`; also return whether it references a place."""
        before = self._place_refs
        ans = self._visit_expr(node)
        return ans, self._place_refs != before

    def _visit_target(self, node: ast.expr, value_refs_places: bool) -> ast.expr:
        with self._stmt_refs_places.bind(value_refs_places):
            return self._visit_expr(node)

    def visit_stmts(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        ans: list[ast.stmt] = []
        for stmt in stmts:
            out = self.visit(stmt)
            if out is None:
                continue
            if isinstance(out, list):
                ans.extend(cast(list[ast.stmt], out))
                continue
            assert isinstance(out, ast.stmt)
            ans.append(out)
        return ans

    @contextmanager
    def _exclude_places(self, names: set[str]) -> Iterator[None]:
        if not names:
            yield None
            return
        with self._places.bind(self._places.value - names):
            yield None

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "comprehension")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        if node.target.id in self._places.value:
            raise _TranspileErrorWrapped(
                node, TranspileNotImplementedError, ":= assigning a non-static local"
            )
        return self._generic_visit_sametype(node)

    #####
    # scopes
    #####

    def _visit_nested_def[N: (ast.FunctionDef, ast.AsyncFunctionDef)](self, node: N) -> N:
        """A nested def is a trace-time python helper; see `transpile`."""
        shadowed = resolve_scope_shadowed_names(node)
        visible_places = self._places.value - shadowed

        _NESTED_DEFS.append(_NestedDef(copy.deepcopy(node), frozenset(visible_places)))
        marker = ast.copy_location(_embed_call(_mark_nested_def, len(_NESTED_DEFS) - 1), node)

        # decorators and defaults are evaluated in the enclosing scope
        node.decorator_list = [self._visit_expr(x) for x in node.decorator_list]
        node.args.defaults = [self._visit_expr(x) for x in node.args.defaults]
        node.args.kw_defaults = [
            None if x is None else self._visit_expr(x) for x in node.args.kw_defaults
        ]

        with self._places.bind(visible_places), self._helper.bind(True):
            node.body = self.visit_stmts(node.body)

        node.decorator_list.append(marker)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._visit_nested_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._visit_nested_def(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        raise _TranspileErrorWrapped(node, TranspileNotImplementedError, "class def")

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        with self._exclude_places(resolve_scope_shadowed_names(node)):
            return self._generic_visit_sametype(node)

    #####
    # local variables
    #####

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self._places.value:
            return node

        self._place_refs += 1
        assert isinstance(node.ctx, (ast.Load, ast.Store, ast.Del))
        expr = ast.Attribute(value=ast.Name(id=node.id, ctx=ast.Load()), attr="value", ctx=node.ctx)
        return expr  # no copy_location: have diagnostic show the containing statement

    def _needs_store_proxy(self, node: ast.Attribute | ast.Subscript, refs_before: int) -> bool:
        # without places involved, a plain setattr/setitem cannot alias or bypass a place
        if not isinstance(node.ctx, (ast.Store, ast.Del)):
            return False
        return self._place_refs != refs_before or self._stmt_refs_places.value

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        before = self._place_refs
        node = self._generic_visit_sametype(node)
        if not self._needs_store_proxy(node, before):
            return node

        place = _embed_call(_attr, node.value, ast.Constant(value=node.attr))
        expr = ast.Attribute(value=place, attr="value", ctx=node.ctx)
        return expr  # no copy_location: have diagnostic show the containing statement

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        before = self._place_refs
        node = self._generic_visit_sametype(node)
        if not self._needs_store_proxy(node, before):
            return node

        place = ast.Subscript(
            value=_embed_call(_subs, node.value), slice=node.slice, ctx=ast.Load()
        )
        expr = ast.Attribute(value=place, attr="value", ctx=node.ctx)
        return expr  # no copy_location: have diagnostic show the containing statement

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node.value, refs = self._visit_value(node.value)
        node.targets = [self._visit_target(x, refs) for x in node.targets]
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        node.value, refs = self._visit_value(node.value)
        target = self._visit_target(node.target, refs)
        assert isinstance(target, (ast.Name, ast.Attribute, ast.Subscript))
        node.target = target
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Annotation is kept only if the target is untouched.

        Rewritten targets are attributes like `x.value`, for which python rejects a "simple"
        `AnnAssign`; annotations of locals are never evaluated, so a plain `Assign` is equivalent.
        """
        orig_target = node.target

        if node.value is None:
            target = self._visit_target(orig_target, False)
            if target is orig_target:
                return node
            # places are initialized in the function prologue
            return ast.copy_location(ast.Pass(), node)

        value, refs = self._visit_value(node.value)
        target = self._visit_target(orig_target, refs)
        if target is orig_target:
            node.value = value
            return node
        return ast.copy_location(ast.Assign(targets=[target], value=value), node)

    def visit_With(self, node: ast.With) -> ast.AST:
        for item in node.items:
            item.context_expr, refs = self._visit_value(item.context_expr)
            if item.optional_vars is not None:
                item.optional_vars = self._visit_target(item.optional_vars, refs)
        node.body = self.visit_stmts(node.body)
        return node

    #####
    # call
    #####

    def visit_Call(self, node: ast.Call) -> ast.AST:
        before = self._place_refs
        node = self._generic_visit_sametype(node)
        if self._place_refs == before:
            return node

        expr = ast.Call(
            func=_embed_object(_call), args=[node.func, *node.args], keywords=node.keywords
        )
        return ast.copy_location(expr, node)

    #####
    # Control flow transformations
    #####

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        """Transform `if x: body else: orelse` into

        ```
        tmp = x
        with _if_body(tmp):  # if_(tmp)
            if tmp is not False:
                body
            if tmp is not True:
                with _if_else(tmp):  # else_()
                    orelse
        ```

        On a runtime `x`, both python `if`s hold: this is `with if_(x): body; with else_():
        orelse`, the else_() called from within the if_() context manager as it must be. On a
        python bool, only the branch taken is traced, like `and`, `or`, `not` and ternaries.
        """

        node = self._generic_visit_sametype(node)
        if self._helper.value:
            return node

        test_name = self._tmp("if_test")

        def test() -> ast.Name:
            return ast.Name(id=test_name, ctx=ast.Load())

        def unless(value: bool, body: list[ast.stmt]) -> ast.If:
            return ast.If(
                test=ast.Compare(left=test(), ops=[ast.IsNot()], comparators=[ast.Constant(value)]),
                body=body,
                orelse=[],
            )

        if_body: list[ast.stmt] = [unless(False, node.body)]
        if node.orelse:
            else_with = ast.With(
                items=[ast.withitem(context_expr=_embed_call(_if_else, test()))],
                body=node.orelse,
            )
            if_body.append(unless(True, [else_with]))

        init_test = ast.Assign(targets=[ast.Name(id=test_name, ctx=ast.Store())], value=node.test)
        if_call = ast.copy_location(_embed_call(_if_body, test()), node.test)
        if_with = ast.With(
            items=[ast.withitem(context_expr=if_call)],
            body=if_body,
        )

        return [ast.copy_location(init_test, node), ast.copy_location(if_with, node)]

    def visit_While(self, node: ast.While) -> ast.AST:
        """Transform `while x:` -> `with while_(lambda: x):`"""

        node = self._generic_visit_sametype(node)
        if self._helper.value:
            return node

        test_lambda = ast.Lambda(args=ast.arguments(), body=node.test)

        if_body = node.body

        if node.orelse:
            else_with = ast.With(
                items=[ast.withitem(context_expr=_embed_call(else_))],
                body=node.orelse,
            )
            if_body.append(else_with)

        # Create: with while_(lambda: test):
        while_with = ast.With(
            items=[
                ast.withitem(context_expr=_embed_call(while_, test_lambda)),
            ],
            body=if_body,
        )

        return ast.copy_location(while_with, node)

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.stmt]:
        """Transform `for x in y:` -> `for x in _patch_for_loop(y):`"""
        node.iter, refs = self._visit_value(node.iter)
        node.target = self._visit_target(node.target, refs)
        node.body = self.visit_stmts(node.body)
        node.orelse = self.visit_stmts(node.orelse)
        if self._helper.value:
            return node

        end_name = self._tmp("for_end")
        end_label = ast.Name(id=end_name, ctx=ast.Load())

        init_end = ast.Assign(
            targets=[ast.Name(id=end_name, ctx=ast.Store())],
            value=_embed_call(mk_internal_label, "for_end"),
        )

        new_for = ast.For(
            target=node.target,
            iter=_embed_call(_patch_for_loop, node.iter, break_to=end_label),
            body=node.body,
        )

        emit_end = ast.Expr(value=_embed_call(label, end_label))

        init_end = ast.copy_location(init_end, node)
        new_for = ast.copy_location(new_for, node)
        emit_end = ast.copy_location(emit_end, node)
        return [init_end, new_for] + node.orelse + [emit_end]

    def visit_Break(self, node: ast.Break) -> ast.AST:
        """Transform `break` -> `break_()`"""
        if self._helper.value:
            return node
        expr = ast.Expr(value=_embed_call(break_))
        return ast.copy_location(expr, node)

    def visit_Continue(self, node: ast.Continue) -> ast.AST:
        """Transform `continue` -> `continue_()`"""
        if self._helper.value:
            return node
        expr = ast.Expr(value=_embed_call(continue_))
        return ast.copy_location(expr, node)

    def visit_Return(self, node: ast.Return) -> ast.AST:
        """Transform `return x` -> `return_(x)`"""
        node = self._generic_visit_sametype(node)
        if self._helper.value:
            return node

        if node.value is not None:
            expr = ast.Expr(value=_embed_call(return_, node.value))
        else:
            expr = ast.Expr(value=_embed_call(return_))
        return ast.copy_location(expr, node)

    #####
    # Expression transformations
    #####

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        """Transform `x if y else z` -> `_ifexp(y, lambda: x, lambda: z)`"""
        node = self._generic_visit_sametype(node)
        if self._helper.value:
            return node

        # Create lambdas for body and orelse
        body_lambda = ast.Lambda(args=ast.arguments(), body=node.body)
        orelse_lambda = ast.Lambda(args=ast.arguments(), body=node.orelse)

        expr = _embed_call(_ifexp, node.test, body_lambda, orelse_lambda)
        return ast.copy_location(expr, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        """Transform `not x` -> `_not(x)`"""
        node = self._generic_visit_sametype(node)

        if isinstance(node.op, ast.Not) and not self._helper.value:
            expr = _embed_call(_not, node.operand)
            return ast.copy_location(expr, node)

        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        """Transform `x and y` -> `_and(x, lambda: y)` and `x or y` -> `_or(x, lambda: y)`"""
        # BoolOp has a list of values for chained operators: a and b and c
        # We need to reduce left-to-right: _and(_and(a, lambda: b), lambda: c)

        node = self._generic_visit_sametype(node)
        if self._helper.value:
            return node

        if isinstance(node.op, ast.And):
            fn = _and
        elif isinstance(node.op, ast.Or):
            fn = _or
        else:
            return node

        # Start with first value, then chain
        result = node.values[0]
        for val in node.values[1:]:
            # Create lambda for right-hand side
            rhs_lambda = ast.Lambda(args=ast.arguments(), body=val)
            result = _embed_call(fn, result, rhs_lambda)

        return ast.copy_location(result, node)


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


def _transpile_ast(
    fn_node: FunctionDefNode,
    eval_expr: EvalExpr,
    param_annotations: dict[str, Any],
    outer_places: frozenset[str],
) -> tuple[FunctionDefNode, frozenset[str]]:
    """Returns the transpiled function and its static params."""
    try:
        return _transpile_ast_impl(fn_node, eval_expr, param_annotations, outer_places)
    except _TranspileErrorWrapped as e:
        # Defer the error to when the function runs, with a traceback pointing at the
        # unsupported syntax / reserved name in the original source.
        fn_node.body = [
            ast.Import(names=[ast.alias(name="sicc", asname=_SICC_IMPORT)]),
            ast.copy_location(ast.Raise(exc=_embed_call(e.err_type, e.msg)), e.node),
        ]
        return fn_node, frozenset()


def _transpile_ast_impl(
    fn_node: FunctionDefNode,
    eval_expr: EvalExpr,
    param_annotations: dict[str, Any],
    outer_places: frozenset[str],
) -> tuple[FunctionDefNode, frozenset[str]]:
    _assert_no_reserved_prefix(fn_node)
    locals_, params = resolve_function_scope(fn_node)
    statics = _resolve_statics(fn_node, locals_ | params, eval_expr, param_annotations)
    own_places = (locals_ | params) - statics
    places = own_places | (outer_places - locals_ - params)

    _check_static_stores(fn_node, places)

    transformer = _TranspileBodyTransformer(places)
    fn_node.body = transformer.visit_stmts(fn_node.body)

    init: list[ast.stmt] = [
        ast.Import(names=[ast.alias(name="sicc", asname=_SICC_IMPORT)]),
        *[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=_embed_call(
                    _init_place, ast.Constant(name), ast.Name(id=name, ctx=ast.Load())
                ),
            )
            for name in sorted(params & own_places)
        ],
        *[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=_embed_call(_init_place, ast.Constant(name)),
            )
            for name in sorted(locals_ & own_places)
        ],
    ]

    for st in init:
        ast.copy_location(st, fn_node)
        # FIXME: without this it doesnt display correctly
        st.end_lineno = st.lineno
        st.end_col_offset = st.col_offset

    insert_at = 0
    if (
        fn_node.body
        and isinstance(fn_node.body[0], ast.Expr)
        and isinstance(fn_node.body[0].value, ast.Constant)
        and isinstance(fn_node.body[0].value.value, str)
    ):
        insert_at = 1

    fn_node.body[insert_at:insert_at] = init
    return fn_node, frozenset(params & statics)


@dataclass(frozen=True)
class _Compiled:
    fn: Callable[..., Any]
    orig_str: str
    transpiled_str: str


@dataclass
class Transpiled[**P, R](CallDescMixin):
    """
    Result of `@transpile`. The function is transpiled and compiled lazily, on first call or
    when its source is requested, so annotations can refer to names defined later.
    """

    orig: Callable[P, R]
    inline: bool

    def __post_init__(self) -> None:
        self.__name__ = self.orig.__name__
        self.__qualname__ = self.orig.__qualname__
        self.__module__ = self.orig.__module__
        self.__wrapped__ = self.orig

    @functools.cached_property
    def _compiled(self) -> _Compiled:
        fn = self.orig
        assert isinstance(fn, types.FunctionType)
        nested = _NESTED_DEF_BY_CODE.get(fn.__code__)

        transpiled_fn, orig_str, transpiled_str, static_params = transpile_runtime(
            fn,
            transform=lambda node, eval_expr: _transpile_ast(
                node,
                eval_expr,
                fn.__annotations__,
                frozenset() if nested is None else nested.outer_places,
            ),
            fn_node=None if nested is None else nested.node,
        )

        wrap_fn: Any
        if inspect.iscoroutinefunction(fn):
            wrap_fn = identity
        else:
            if static_params and not self.inline:
                raise TypeError(
                    f"@transpile function {fn.__qualname__!r} has Static params "
                    f"{sorted(static_params)}; these require @transpile(inline=True), "
                    "since a subroutine receives its arguments as runtime variables"
                )
            if inspect.isgeneratorfunction(fn):
                wrap_fn = shared_generator(inline=self.inline)
            else:
                wrap_fn = subr(inline=self.inline)

        return _Compiled(
            fn=wrap_fn(transpiled_fn), orig_str=orig_str, transpiled_str=transpiled_str
        )

    @property
    def transpiled(self) -> Callable[P, R]:
        return self._compiled.fn

    @property
    def orig_str(self) -> str:
        return self._compiled.orig_str

    @property
    def transpiled_str(self) -> str:
        return self._compiled.transpiled_str

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


@allow_curry
def transpile[**P, R](fn: Callable[P, R], /, *, wrap: bool = True, inline: bool = False):
    """
    transform a function to be one suitable for tracing. designed to be transparent.

    transpiling is lazy: it happens on first call, or when the source is requested
    (`str(fn)`, `fn.pretty()`, `sicc <file> transpile`). annotations can therefore refer to
    names defined later in the module.

    **printed output**
    `rich.print(transpiled_fn.pretty())` prints the transpiled source. this is the code that will be executed
    (conceptually as if it replaced the original function); it is compiled through an internal
    factory and then rebound to `fn`'s globals, closure cells and defaults.

    **local variables**
    non-static locals and parameters are rewritten to `_Place` instances.
    reads/writes go through the `.value` property:
    - `a` -> `a.value`
    - `a.b` -> `a.value.b`
    - `a[b]` -> `a.value[b]`

    attribute/item stores are wrapped to preserve structural assignment semantics:
    - `a.b = x` -> `_attr(a.value, "b").value = x`
    - `a[b] = x` -> `_subs(a.value)[b].value = x`

    calls are wrapped to reject calls that modify their arguments (which would bypass `_Place`):
    - `f(a)` -> `_call(f, a.value)`

    **untouched code**
    `_call`, `_attr` and `_subs` only apply where a non-static local is involved:
    - a call is wrapped only if its callee or arguments reference a non-static local;
      `something()`, `d.foo(1)` stay as is
    - an attribute/item store is wrapped only if its statement (target or value) references a
      non-static local; `d.On = True` stays as is
    control flow and `not`/`and`/`or`/ternary are always rewritten, since python's versions
    call `bool()`, which runtime values do not support.

    **static locals**
    a local or parameter annotated `Static[T]` (any annotation carrying the `Static` marker in
    its `Annotated` metadata, e.g. `Annotated[Static[int], ...]` or an alias of it) is a plain
    trace-time python variable, never rewritten:
    ```
    i: Static[int]
    for i in range(3):  # i is a python int
        ...
    seen: Static[list[Int]] = []
    seen.append(x)  # appends at trace time
    ```
    names bound by `def`, `class` and `import` are static as well.
    - annotations of locals and parameters are evaluated (in `fn`'s globals and closure) when
      transpiling; an annotation that fails to evaluate raises `TranspileError`
    - a name with both a static and a non-static annotation raises `TranspileError`
    - static parameters receive the caller's python value; they require `inline=True` (or an
      `async def`) since a subroutine receives its arguments as runtime variables
    - assigning to a static (or `nonlocal`/`global`) name under a runtime `if`/`while`/`and`/
      `or`/ternary raises `TranspileError`: `if_` traces both branches and `while_` traces its body
      once, so the assignment would run once at trace time regardless of the condition. `for`
      loops are unrolled at trace time and allow it.

    **non-local lookups**
    variables that are not locals to `fn` under @transpile are kept as is. this includes:
    (1) vars that refers to outside `fn`
    (2) vars in inner scope like args or local vars of inner def/lambda, including those shadowing locals of `fn`
    `nonlocal` stays; inner references are recognized as refereing the outer one and therefore rewritten.

    **nested def**
    a `def` inside `fn` is a trace-time python helper:
    - its body is not transpiled: `if`, `while`, `return`, `and`, ... are python's, its own
      locals are plain python variables
    - references to non-static locals of `fn` are rewritten as in `fn` (including `_call`,
      `_attr`, `_subs`), so reading and writing them (through `nonlocal`) works
    - decorators and default values are evaluated in `fn` and follow `fn`'s rules
    applying `@transpile` to a nested def transpiles it as usual; non-static locals of `fn` it
    references are accessed as places through the closure:
    ```
    acc = 0

    @transpile
    def add(v: Int):
        nonlocal acc
        if v > 0:  # runtime if
            acc += v
    ```

    **control flow**
    `if x: ... else: ...` -> `with if_(x): ... with else_(): ...` (on a python bool, at trace time)
    `x if y else z` -> `_ifexp(y, lambda: x, lambda: z)` (on a python bool, at trace time)
    `while x` -> `with while_(lambda: x): ... with else_(): ...`
    `for x in y` -> `for x in _patch_for_loop(y, ...)`
    `break` -> `break_`
    `continue` -> `continue_`
    `return` -> `return_`
    `del x` -> `del x.value` (implemented as set to undef)

    **expression**
    `not x` -> `_not(x)`
    `x and y` -> `_and(x, lambda: y)`
    `x or y` -> `_or(x, lambda: y)`

    **not supported (raises TranspileNotImplementedError)**
    comprehensions, `class` definitions, `:=` assigning a non-static local

    **planned**
    `match` -> as expected
    `:=` on non-static locals -> as expected
    `global` -> `_Place` object stored in global scope

    **all other contstructs**
    kept as is

    **note on unsupported syntax**
    unsupported constructs are rewritten into a runtime `raise`, producing a useful
    backtrace pointing at the original source location when executed.
    """

    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"@transpile can only be applied to functions, got {type(fn)!r}")

    return Transpiled[P, R](orig=fn, inline=inline)
