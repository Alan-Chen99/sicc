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


def _rewrite_closure_cells(
    fn_node: FunctionDefNode, closure: _Closure
) -> tuple[FunctionDefNode, _Closure]:
    """Rewrite freevar reads to access the underlying closure cell at runtime.

    This keeps transpiled functions observing freevar updates (rebinding) after `@transpile`
    runs, matching normal Python closure semantics.
    """
    if not closure:
        return fn_node, closure

    used_names: set[str] = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.arg):
            used_names.add(node.arg)

    name_to_cellvar: dict[str, str] = {}
    rewritten_closure: _Closure = []

    for name, cell in closure:
        base = f"__sicc_closure_cell__{name}"
        cellvar = base
        counter = 0
        while cellvar in used_names:
            counter += 1
            cellvar = f"{base}_{counter}"
        used_names.add(cellvar)

        name_to_cellvar[name] = cellvar
        rewritten_closure.append((cellvar, cell))

    class _ClosureCellContentsTransformer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if (
                isinstance(node.ctx, ast.Load)
                and (cellvar := name_to_cellvar.get(node.id)) is not None
            ):
                expr = ast.Attribute(
                    value=ast.Name(id=cellvar, ctx=ast.Load()),
                    attr="cell_contents",
                    ctx=ast.Load(),
                )
                return ast.copy_location(expr, node)
            return node

    fn_node_out = cast(FunctionDefNode, _ClosureCellContentsTransformer().visit(fn_node))
    return fn_node_out, rewritten_closure


def _compile_function_def(
    fn_node: FunctionDefNode,
    closure: _Closure,
    *,
    filename: str,
    scope: dict[str, Any],
) -> types.FunctionType:
    """Compile `fn_node` as a function, binding `closure` values via a generated factory."""
    factory_name = f"__factory_{fn_node.name}_{id(fn_node)}"
    fn_node_exec = copy.deepcopy(fn_node)

    freevars = [name for name, _ in closure]

    factory_def = ast.FunctionDef(
        name=factory_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in freevars],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=[
            fn_node_exec,
            ast.Return(value=ast.Name(id=fn_node.name, ctx=ast.Load())),
        ],
        decorator_list=[],
    )

    factory_mod = ast.Module(body=[factory_def], type_ignores=[])
    ast.fix_missing_locations(factory_mod)

    # print(ast.unparse(factory_mod))

    code_obj = compile(factory_mod, filename=filename, mode="exec")
    namespace: dict[str, Any] = {}
    exec(code_obj, scope, namespace)

    factory = namespace[factory_name]
    freevar_values = [value for _, value in closure]
    return factory(*freevar_values)


def transpile_runtime[**P, R](
    fn: Callable[P, R],
    *,
    transform: Callable[[FunctionDefNode], FunctionDefNode],
) -> tuple[Callable[P, R], str, str]:
    """Transpile and compile `fn`, returning `(callable, original_source, transpiled_source)`."""
    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"@transpile can only be applied to functions, got {type(fn)!r}")

    source_original, starting_line, indent = _get_dedented_source(fn)
    tree = ast.parse(source_original)

    original_fn_node = _find_function_def(tree, fn.__name__)

    # When `@transpile` is used alongside other decorators, the source returned by inspect
    # includes those decorator expressions. Recompiling them would re-apply `@transpile`
    # and friends, leading to recursion at import-time. Only compile the bare function.
    fn_node = copy.deepcopy(original_fn_node)
    fn_node.decorator_list = []

    transpiled_node = transform(fn_node)

    closure = _get_closure(fn)
    transpiled_node, closure = _rewrite_closure_cells(transpiled_node, closure)

    tree.body = [transpiled_node]
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, starting_line - 1)
    _increment_col_offset(tree, len(indent))

    source_transpiled = ast.unparse(ast.Module(body=[transpiled_node], type_ignores=[]))

    transpiled_fn = _compile_function_def(
        transpiled_node,
        closure,
        filename=fn.__code__.co_filename,
        scope=fn.__globals__,
    )

    transpiled_fn.__qualname__ = fn.__qualname__

    return cast(Callable[P, R], transpiled_fn), source_original, source_transpiled


def resolve_function_scope(fn_node: FunctionDefNode) -> tuple[set[str], set[str]]:
    """Return `(locals, parameters)` for `fn_node`'s scope."""
    code = ast.unparse(ast.Module(body=[fn_node], type_ignores=[]))
    mod_table = symtable.symtable(code, filename="<transpile>", compile_type="exec")

    fn_table: symtable.Function | None = None
    for child in mod_table.get_children():
        if child.get_name() == fn_node.name:
            if not isinstance(child, symtable.Function):
                raise TypeError(f"Expected {fn_node.name!r} to be a function, got {type(child)!r}")
            fn_table = child
            break
    if fn_table is None:
        raise ValueError(f"Expected to find function table for {fn_node.name!r}")

    params = set(fn_table.get_parameters())
    locals_ = set(fn_table.get_locals()) - params
    return locals_, params


def resolve_scope_shadowed_names(node: ScopeNode) -> set[str]:
    """Return names local to `node`'s scope that shadow outer names."""

    def _symtable_for(node: ScopeNode) -> symtable.SymbolTable:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                code = ast.unparse(ast.Module(body=[node], type_ignores=[]))
                mod_table = symtable.symtable(code, filename="<transpile>", compile_type="exec")
                for child in mod_table.get_children():
                    if (
                        isinstance(child, (symtable.Function, symtable.Class))
                        and child.get_name() == node.name
                    ):
                        return child
                raise ValueError(f"Expected to find scope table for {node.name!r}")
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
    if isinstance(table, symtable.Function):
        return set(table.get_locals()) | set(table.get_globals())

    assert isinstance(table, symtable.Class)
    return {
        sym.get_name() for sym in table.get_symbols() if sym.is_local() or sym.is_declared_global()
    }


def resolve_function_locals(fn_node: FunctionDefNode) -> set[str]:
    """Return names local to `fn_node`'s scope, excluding its parameters."""
    locals_, _ = resolve_function_scope(fn_node)
    return locals_
