from __future__ import annotations

import dataclasses
import functools
import inspect
from dataclasses import dataclass
from typing import Annotated
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Coroutine
from typing import TypeVar
from typing import cast
from typing import get_args
from typing import get_origin
from typing import get_type_hints

import optree
import rich.repr
from rich import print as print  # autoflake: skip

from ._api import Variable
from ._api import read_uservalue
from ._control_flow import BlockRef
from ._control_flow import block
from ._control_flow import clear_control_flow_hooks
from ._core import Label
from ._core import MVar
from ._core import VarT
from ._core import get_type
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import describe_fn
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._instructions import jump
from ._tracing import RawSubr
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import mk_mvar
from ._tracing import trace_to_raw_subr
from ._tree_utils import PyTree
from ._tree_utils import TreeSpec
from ._tree_utils import ValSpec
from ._tree_utils import pytree
from ._tree_utils import tree_flatten
from ._utils import CallDescMixin
from ._utils import Cell
from ._utils import allow_curry
from ._utils import normalize_function_args
from .config import with_status

register_exclusion(__file__)

################################################################################
# handling return; `inline_subr`
################################################################################


# a dummy type annotation
class _Function[R]:
    pass


# annotation for return type of subr
# if annotated, static type checkers understand the return type at callsite
type FunctionRet[R = Any] = R | _Function[R] | None

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
            assert not isinstance(ans, _Function)

            if ans is not None:
                b.break_(ans)
            elif len(b._break_paths) == 0:
                b.break_(None)  # pyright: ignore[reportArgumentType]

        return b.value

    return inner


def inline_subr_async[**P, R](
    fn: Callable[P, Awaitable[FunctionRet[R]]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """`inline_subr` for a coroutine function; its bindings stay active across awaits."""

    async def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with (
            block(BlockRef[R]) as b,
            clear_control_flow_hooks(),
            _RETURN_HOOK.bind(b.break_),
        ):
            ans = await fn(*args, **kwargs)
            assert not isinstance(ans, _Function)

            if ans is not None:
                b.break_(ans)
            elif len(b._break_paths) == 0:
                b.break_(None)  # pyright: ignore[reportArgumentType]

        return b.value

    return inner


################################################################################
# subr tracing and calling
################################################################################


def trace_to_subr[A, R](
    fn: Callable[[A], FunctionRet[R]],
    arg_tree: TreeSpec[A],  # tree spec of fn
) -> TracedSubr[A, R]:
    # print(f"arg_tree {describe_fn(fn)}:", arg_tree)

    ret_tree: Cell[TreeSpec[R]] = Cell()

    ra_mvar = mk_mvar(Label)
    # arg_mvars = tuple(mk_mvar(x.typ) for x in arg_tree.leaves)

    arg_variables = [Variable(spec.type, _read_only=not spec.mutable) for spec in arg_tree.leaves]

    start = mk_internal_label(f"[{describe_fn(fn)}]", private=False)

    with with_status(describe_fn(fn)), trace_to_raw_subr() as raw_subr:
        with add_debug_info(DebugInfo(describe=f"<function {describe_fn(fn)}>")):
            label(start)

        # allow spilling ra into another reg at function start
        ra_mvar_inner = mk_mvar(Label)
        ra_mvar_inner.write(ra_mvar.read())

        # Explicitly read from arg MVars to establish data flow for register allocation
        # This mirrors the old behavior where args were read at function start
        with add_debug_info(DebugInfo(describe=f"args of {describe_fn(fn)}")):
            for v in arg_variables:
                v._inner.write(v._inner.read())

        arg_obj = arg_tree.unflatten(arg_variables, check_mut=True)
        ret_obj = inline_subr(fn)(arg_obj)
        out_values, ret_tree.value = TreeSpec.flatten(ret_obj)
        ret_tree.value = TreeSpec(
            ret_tree.value.tree,
            [ValSpec(type=leaf.type, mutable=False) for leaf in ret_tree.value.leaves],
        )

        ret_vals = [read_uservalue(x) for x in out_values]
        ret_mvars = tuple(mk_mvar(get_type(x)) for x in ret_vals)

        for mv, v in zip(ret_mvars, ret_vals):
            with add_debug_info(DebugInfo(describe=f"return value from {describe_fn(fn)}")):
                mv.write(v)

        with add_debug_info(DebugInfo(describe=f"return from {describe_fn(fn)}")):
            jump(ra_mvar_inner.read())

    return TracedSubr(
        raw_subr=raw_subr.value,
        arg_tree=arg_tree,
        ret_tree=ret_tree.value,
        start=start,
        ra_mvar=ra_mvar,
        arg_mvars=tuple(x._inner for x in arg_variables),
        ret_mvars=ret_mvars,
    )


@dataclass
class TracedSubr[A, R]:
    raw_subr: RawSubr
    arg_tree: TreeSpec[A]
    ret_tree: TreeSpec[R]

    start: Label
    ra_mvar: MVar[Label]
    arg_mvars: tuple[MVar, ...]
    ret_mvars: tuple[MVar, ...]

    def call(self, arg: A) -> R:
        self.raw_subr.record_call()

        # note: compiler cannot handle undef effectively yet

        # set args
        values = self.arg_tree.flatten_up_to(arg, check_mut=True)
        for mv, v, spec in zip(self.arg_mvars, values, self.arg_tree.leaves):
            mv.write(read_uservalue(v))
            # if spec.mutable:
            #     assert isinstance(v, Variable)
            #     v.value = undef()  # otherwise compiler may fail to prove that mv and v may alias

        # call
        l = mk_internal_label("call_ret")
        self.ra_mvar.write(l)
        with track_caller():
            jump(self.start)
        label(l)

        # return vars
        ret_obj = self.ret_tree._unflatten_vals_ro(x.read() for x in self.ret_mvars)

        # writing back mutable args
        for mv, v, spec in zip(self.arg_mvars, values, self.arg_tree.leaves):
            if spec.mutable:
                assert isinstance(v, Variable)
                v._inner.write(mv.read())

        # # clear mvars
        # for mv in self.arg_mvars:
        #     mv.write(Undef(mv.type))
        # for mv in self.ret_mvars:
        #     mv.write(Undef(mv.type))

        return ret_obj


################################################################################
# annotation handling, API
################################################################################


@dataclass(frozen=True)
class Arg:
    type: PyTree[type[VarT] | Arg | None] | Any = None
    _ = dataclasses.KW_ONLY

    mutable: bool = False

    def normalize(self) -> PyTree[_SubtreeSpec]:
        return _normalize_arg(self.type, self.mutable)


def _normalize_arg(val: Any, mutable: bool) -> PyTree[_SubtreeSpec]:
    """Normalize a value/type specification into a PyTree of _SubtreeSpec."""
    if val is None:
        # No type specified - will be resolved at call time
        return _SubtreeSpec(_type=None, _mutable=mutable)

    if isinstance(val, Arg):
        # Nested Arg - normalize recursively, combine mutable flags
        result = val.normalize()
        if mutable and not val.mutable:
            # Outer mutable overrides inner non-mutable
            return _apply_mutable(result)
        return result

    if isinstance(val, type):
        if issubclass(val, VarT):
            return _SubtreeSpec(_type=val, _mutable=mutable)
        # For dataclass types, return a spec that will be resolved at runtime
        return _SubtreeSpec(_type=None, _mutable=mutable)

    # Try to flatten as pytree - if it has children, process recursively
    # Use none_is_leaf=True to treat None as leaves
    leaves, tree = optree.tree_flatten(val, none_is_leaf=True, namespace="sicc._tree_utils")
    if len(leaves) == 1 and leaves[0] is val:
        # A non-pytree leaf (e.g. an int) would recurse forever; treat as runtime-resolved.
        return _SubtreeSpec(_type=None, _mutable=mutable)
    if len(leaves) == 0:
        # Scalar or empty - treat as runtime resolved
        return _SubtreeSpec(_type=None, _mutable=mutable)

    # Process each leaf recursively
    new_leaves = [_normalize_arg(leaf, mutable) for leaf in leaves]
    return cast(PyTree[_SubtreeSpec], tree.unflatten(new_leaves))


def _apply_mutable(tree: PyTree[_SubtreeSpec]) -> PyTree[_SubtreeSpec]:
    """Apply mutable=True to all _SubtreeSpec leaves in the tree."""

    def make_mutable(spec: _SubtreeSpec) -> _SubtreeSpec:
        return _SubtreeSpec(_type=spec._type, _mutable=True)

    return cast(
        PyTree[_SubtreeSpec],
        optree.tree_map(
            make_mutable,
            cast(optree.PyTree[_SubtreeSpec], tree),
            is_leaf=lambda x: isinstance(
                x, _SubtreeSpec
            ),  # pyright: ignore[reportUnnecessaryIsInstance]
        ),
    )


def _annotation_to_arg(annotation: Any) -> Arg:
    """Convert a type annotation to an Arg specification."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        # No annotation or Any - default behavior
        return Arg()

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Handle Annotated[T, Arg(...), ...] - may have multiple Args (e.g., from Mutable[Annotated[...]])
    if origin is Annotated:
        base_type = args[0]
        # Combine all Arg metadata
        arg_type = None
        arg_mutable = False
        for metadata in args[1:]:
            if isinstance(metadata, Arg):
                if arg_type is None and metadata.type is not None:
                    arg_type = metadata.type
                if metadata.mutable:
                    arg_mutable = True
            elif arg_type is None:
                # Allow pytree-shaped metadata (e.g. dataclass schemas containing `Arg` leaves).
                arg_type = metadata

        # Get the actual type from annotation (including Variable extraction)
        inner_arg = _annotation_to_arg(base_type)

        # Merge: explicit arg_spec overrides inner_arg where specified
        final_type = arg_type if arg_type is not None else inner_arg.type
        final_mutable = arg_mutable or inner_arg.mutable
        return Arg(type=final_type, mutable=final_mutable)

    # Handle Variable[T]
    if origin == Variable:
        inner_type = args[0] if args else None
        if inner_type is not None and isinstance(inner_type, type) and issubclass(inner_type, VarT):
            return Arg(type=inner_type, mutable=False)
        return Arg()

    # Handle bare types
    if isinstance(annotation, type):
        if issubclass(annotation, VarT):
            return Arg(type=annotation, mutable=False)
        # Class type (like Data) - will be resolved at runtime
        return Arg()

    return Arg()


def get_argspec_from_annotations(fn: Any) -> PyTree[Arg]:
    """Extract argument specification from function annotations.

    Returns a PyTree[Arg] representing ((positional_args...), {keyword_args...}).
    For functions with *args or **kwargs, returns Arg() to indicate runtime inference.
    """
    sig = inspect.signature(fn)

    # Get type hints with extras to preserve Annotated info
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    positional_args: list[Arg] = []
    keyword_args: dict[str, Arg] = {}

    for name, param in sig.parameters.items():
        # Functions with *args or **kwargs cannot have static argspec - use runtime inference
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return Arg()

        annotation = hints.get(name, param.annotation)
        arg = _annotation_to_arg(annotation)

        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_args.append(arg)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword_args[name] = arg

    return (tuple(positional_args), keyword_args)


#####


@dataclass(frozen=True)
class _SubtreeSpec:
    # normalized Arg. still represents a subtree.
    _type: type[VarT] | None = None
    _mutable: bool = False

    def __repr__(self) -> str:
        if self._type is None:
            if self._mutable:
                return "mut"
            return "..."

        if not self._mutable:
            return self._type.__name__
        return f"{self._type.__name__}:mut"

    def _resolve[T](self, val: T) -> TreeSpec[T]:
        if self._type != None:
            return cast(TreeSpec[T], TreeSpec.primitive(self._type, self._mutable))

        _, spec = TreeSpec.flatten(val)
        # Apply the mutable flag to all leaves
        new_leaves = [ValSpec(type=l.type, mutable=self._mutable) for l in spec.leaves]
        return TreeSpec(spec.tree, new_leaves)


T = TypeVar("T")
Mutable = Annotated[T, Arg(mutable=True)]


@dataclass(kw_only=True, frozen=True)
class ArgSpec[T = Any]:
    _prefix_tree: pytree.PyTreeSpec
    _subtree_specs: list[_SubtreeSpec]  # prefix tree with leaves that are Arg
    _resolved: Cell[TreeSpec[T]]

    def __rich_repr__(self) -> rich.repr.Result:
        yield None, self._prefix_tree.unflatten(self._subtree_specs)

    @staticmethod
    def from_args(args: PyTree[Arg]) -> ArgSpec:
        specs, prefix_tree = tree_flatten(Arg(args).normalize())
        return ArgSpec(_prefix_tree=prefix_tree, _subtree_specs=specs, _resolved=Cell())

    def resolve(self, val: T) -> TreeSpec[T]:
        # print(self)

        subtrees = self._prefix_tree.flatten_up_to(cast(optree.PyTree[Any], val))

        # resolve each subtree
        subtree_specs = [
            spec._resolve(subtree) for subtree, spec in zip(subtrees, self._subtree_specs)
        ]

        schema = self._prefix_tree.unflatten([x.as_schema() for x in subtree_specs])
        return TreeSpec.from_schema(schema)  # pyright: ignore[reportReturnType]


F_co = TypeVar("F_co", covariant=True)


@dataclass(kw_only=True)
class Subr[**P, R](CallDescMixin):
    _fn: Callable[P, FunctionRet[R]]
    _arg_spec: ArgSpec
    _inline: bool

    _subr: TracedSubr[Any, Any] | None = None

    def __repr__(self):
        return repr(self._fn)

    def __post_init__(self):
        self.__qualname__ = self._fn.__qualname__
        self.__wrapped__ = self._fn

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if self._inline:
            return inline_subr(self._fn)(*args, **kwargs)

        args, kwargs = normalize_function_args(self._fn, args, kwargs)

        @functools.wraps(self._fn)
        def inner(args_kwargs: tuple[tuple[Any, ...], dict[str, Any]]) -> FunctionRet[R]:
            args, kwargs = args_kwargs
            return self._fn(*args, **kwargs)

        if self._subr is None:
            arg_tree = self._arg_spec.resolve((args, kwargs))
            self._subr = trace_to_subr(inner, arg_tree)

        return self._subr.call((args, kwargs))


@allow_curry
def subr[**P, R](
    fn: Callable[P, FunctionRet[R]],
    /,
    *,
    inline: bool = False,
    arg_spec: ArgSpec | None = None,
) -> Subr[P, R]:

    if arg_spec is None:
        arg_spec = ArgSpec.from_args(get_argspec_from_annotations(fn))

    return Subr(_fn=fn, _arg_spec=arg_spec, _inline=inline)
