from __future__ import annotations

import ast
import copy
import inspect
import symtable
import textwrap
import types
from collections.abc import Callable
from typing import Any
from typing import cast

FunctionDefNode = ast.FunctionDef | ast.AsyncFunctionDef
type _Closure = list[tuple[str, types.CellType]]
ScopeNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda | ast.GeneratorExp


def _find_function_def(tree: ast.Module, name: str) -> FunctionDefNode:
    """Find the top-level function definition named `name` in `tree`."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise ValueError(f"Expected to find function {name!r} in parsed source")


def _leading_indent(lines: list[str]) -> str:
    """Return the indentation prefix for the first non-empty line in `lines`."""
    for line in lines:
        if line.strip() != "":
            return line[: len(line) - len(line.lstrip())]
    return ""


def _increment_col_offset(tree: ast.AST, delta: int) -> None:
    """Add `delta` to `col_offset` and `end_col_offset` for every node in `tree`."""
    if delta == 0:
        return
    for node in ast.walk(tree):
        if (col := getattr(node, "col_offset", None)) is not None:
            setattr(node, "col_offset", col + delta)
        if (col := getattr(node, "end_col_offset", None)) is not None:
            setattr(node, "end_col_offset", col + delta)


def _get_dedented_source(fn: types.FunctionType) -> tuple[str, int, str]:
    """Return the dedented source text and its location metadata for `fn`.

    Returns:
        (source, starting_line, leading_indent)
    """
    source_lines, starting_line = inspect.getsourcelines(fn)
    indent = _leading_indent(source_lines)
    source = textwrap.dedent("".join(source_lines))
    return source, starting_line, indent


def _get_closure(fn: types.FunctionType) -> _Closure:
    """Return `(name, cell)` pairs for `fn`'s closure freevars."""
    if fn.__closure__ is None:
        return []
    return list(zip(fn.__code__.co_freevars, fn.__closure__))


def _eval_in_scope(expr: ast.expr, *, fn: types.FunctionType, closure: _Closure) -> Any:
    """Evaluate `expr` as if it appeared in `fn`'s body, seeing its globals and closure."""
    cells = dict(closure)
    ns: dict[str, Any] = {}
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and (cell := cells.get(node.id)) is not None:
            try:
                ns[node.id] = cell.cell_contents
            except ValueError:
                raise NameError(
                    f"free variable {node.id!r} referenced before assignment in enclosing scope"
                ) from None

    expr_node = ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(expr)))
    code = compile(expr_node, filename=fn.__code__.co_filename, mode="eval", dont_inherit=True)
    return eval(code, fn.__globals__, ns)


def _compile_function_def(
    fn_node: FunctionDefNode,
    orig: types.FunctionType,
    closure: _Closure,
) -> types.FunctionType:
    """Compile `fn_node` into a function sharing `orig`'s globals, closure cells and defaults.

    Sharing the cells (instead of copying values) keeps rebinding and `nonlocal` stores visible
    in both directions, and lets Python's own scoping handle names shadowed by nested scopes.
    """
    factory_name = f"__factory_{fn_node.name}_{id(fn_node)}"
    fn_node_exec = copy.deepcopy(fn_node)
    # defaults come from `orig`, which already evaluated them in the right scope
    fn_node_exec.args.defaults = []
    fn_node_exec.args.kw_defaults = [None] * len(fn_node_exec.args.kwonlyargs)

    # binding the freevars in the factory makes them freevars of the compiled function
    bind_freevars: list[ast.stmt] = [
        ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(None))
        for name, _ in closure
    ]
    factory_def = ast.FunctionDef(
        name=factory_name,
        args=ast.arguments(),
        body=[
            *bind_freevars,
            fn_node_exec,
            ast.Return(value=ast.Name(id=fn_node.name, ctx=ast.Load())),
        ],
        decorator_list=[],
    )

    factory_mod = ast.Module(body=[factory_def], type_ignores=[])
    ast.fix_missing_locations(factory_mod)

    code_obj = compile(factory_mod, filename=orig.__code__.co_filename, mode="exec")
    namespace: dict[str, Any] = {}
    exec(code_obj, orig.__globals__, namespace)
    proto = namespace[factory_name]()
    assert isinstance(proto, types.FunctionType)

    cells = dict(closure)
    ans = types.FunctionType(
        proto.__code__,
        orig.__globals__,
        proto.__name__,
        orig.__defaults__,
        tuple(cells[name] for name in proto.__code__.co_freevars),
    )
    ans.__kwdefaults__ = orig.__kwdefaults__
    ans.__annotations__ = orig.__annotations__
    return ans


type EvalExpr = Callable[[ast.expr], Any]


def transpile_runtime[**P, R, I](
    fn: Callable[P, R],
    *,
    transform: Callable[[FunctionDefNode, EvalExpr], tuple[FunctionDefNode, I]],
    fn_node: FunctionDefNode | None = None,
) -> tuple[Callable[P, R], str, str, I]:
    """Transpile and compile `fn`.

    `fn_node`: AST of `fn` with absolute source positions; parsed from `fn`'s source if None.
    `transform` receives a copy of the AST and an evaluator for expressions in `fn`'s scope.

    Returns `(callable, original_source, transpiled_source, transform_info)`.
    """
    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"@transpile can only be applied to functions, got {type(fn)!r}")

    source_original, starting_line, indent = _get_dedented_source(fn)

    if fn_node is None:
        tree = ast.parse(source_original)
        ast.increment_lineno(tree, starting_line - 1)
        _increment_col_offset(tree, len(indent))
        fn_node = _find_function_def(tree, fn.__name__)

    # When `@transpile` is used alongside other decorators, the source returned by inspect
    # includes those decorator expressions. Recompiling them would re-apply `@transpile`
    # and friends, leading to recursion. Only compile the bare function.
    fn_node = copy.deepcopy(fn_node)
    fn_node.decorator_list = []

    closure = _get_closure(fn)

    def eval_expr(expr: ast.expr) -> Any:
        return _eval_in_scope(expr, fn=fn, closure=closure)

    transpiled_node, info = transform(fn_node, eval_expr)
    ast.fix_missing_locations(transpiled_node)

    source_transpiled = ast.unparse(ast.Module(body=[transpiled_node], type_ignores=[]))

    transpiled_fn = _compile_function_def(transpiled_node, fn, closure)
    transpiled_fn.__qualname__ = fn.__qualname__

    return cast(Callable[P, R], transpiled_fn), source_original, source_transpiled, info


def _def_symtable(node: FunctionDefNode | ast.ClassDef) -> symtable.SymbolTable:
    """Symbol table of a standalone `def`/`class` node."""
    nonlocals = sorted(
        {name for sub in ast.walk(node) if isinstance(sub, ast.Nonlocal) for name in sub.names}
    )
    if not nonlocals:
        code = ast.unparse(ast.Module(body=[node], type_ignores=[]))
        parent = symtable.symtable(code, filename="<transpile>", compile_type="exec")
    else:
        # standalone `nonlocal x` is a SyntaxError; bind the names in a dummy enclosing function
        wrapper = ast.FunctionDef(
            name="__sicc_scope__",
            args=ast.arguments(),
            body=[
                *[
                    ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(None))
                    for name in nonlocals
                ],
                node,
            ],
            decorator_list=[],
        )
        # ast.unparse reads `lineno` of defs and assignments
        wrapper_mod = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        code = ast.unparse(wrapper_mod)
        mod_table = symtable.symtable(code, filename="<transpile>", compile_type="exec")
        (parent,) = mod_table.get_children()

    for child in parent.get_children():
        if child.get_name() == node.name:
            return child
    raise ValueError(f"Expected to find scope table for {node.name!r}")


def resolve_function_scope(fn_node: FunctionDefNode) -> tuple[set[str], set[str]]:
    """Return `(locals, parameters)` for `fn_node`'s scope."""
    fn_table = _def_symtable(fn_node)
    if not isinstance(fn_table, symtable.Function):
        raise TypeError(f"Expected {fn_node.name!r} to be a function, got {type(fn_table)!r}")

    params = set(fn_table.get_parameters())
    locals_ = set(fn_table.get_locals()) - params
    return locals_, params


def resolve_scope_shadowed_names(node: ScopeNode) -> set[str]:
    """Return names local to `node`'s scope that shadow outer names."""

    def _symtable_for(node: ScopeNode) -> symtable.SymbolTable:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                return _def_symtable(node)
            case ast.Lambda():
                code = ast.unparse(ast.Module(body=[ast.Expr(value=node)], type_ignores=[]))
                mod_table = symtable.symtable(code, filename="<transpile>", compile_type="exec")
                for child in mod_table.get_children():
                    if child.get_name() == "lambda":
                        return child
                raise ValueError("Expected to find scope table for lambda")
            case ast.GeneratorExp():
                code = ast.unparse(ast.Module(body=[ast.Expr(value=node)], type_ignores=[]))
                mod_table = symtable.symtable(code, filename="<transpile>", compile_type="exec")
                for child in mod_table.get_children():
                    if child.get_name() == "genexpr":
                        return child
                raise ValueError("Expected to find scope table for genexpr")

    table = _symtable_for(node)
    # standalone, free names of `node` look like implicit globals; only declared ones shadow
    declared_globals = {sym.get_name() for sym in table.get_symbols() if sym.is_declared_global()}
    if isinstance(table, symtable.Function):
        return set(table.get_locals()) | declared_globals

    assert isinstance(table, symtable.Class)
    return {sym.get_name() for sym in table.get_symbols() if sym.is_local()} | declared_globals


def resolve_function_locals(fn_node: FunctionDefNode) -> set[str]:
    """Return names local to `fn_node`'s scope, excluding its parameters."""
    locals_, _ = resolve_function_scope(fn_node)
    return locals_
