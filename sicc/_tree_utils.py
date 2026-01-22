from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import cast

import optree.treespec
import rich.repr
from optree.pytree import reexport
from rich import print as print  # autoflake: skip

from ._api import UserValue
from ._api import Variable
from ._api import VarRead
from ._api import read_uservalue
from ._core import MVar
from ._core import Value
from ._core import VarT
from ._core import VarTS
from ._core import can_cast_implicit
from ._core import promote_types
from ._utils import cast_unchecked


def _dataclass_impl(cls: type[Any] | None = None, /, **kwargs: Any):
    kwargs.setdefault("frozen", True)
    if cls is None:
        return dataclasses.dataclass(**kwargs)
    return dataclasses.dataclass(cls, **kwargs)


if TYPE_CHECKING:
    from dataclasses import dataclass as optree_dataclass_typing

    optree_dataclass = optree_dataclass_typing

    from optree import dataclasses as dataclasses
    from optree import pytree as pytree

    field = dataclasses.field  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
else:
    pytree = reexport(namespace=__name__)
    dataclasses = pytree.dataclasses
    optree_dataclass = _dataclass_impl
    field = dataclasses.field

type PyTree[T] = T | Sequence[PyTree[T]] | Mapping[Any, PyTree[T]] | optree.PyTree[T]

type _BoundMethodCtx = (
    tuple[Literal["unbound"], Any]
    | tuple[Literal["func"], Any, Any]
    | tuple[Literal["name"], str, Any]
    | tuple[Literal["obj"], Any]
)


def _flatten_bound_method(obj: Any) -> tuple[tuple[Any, ...], _BoundMethodCtx]:
    self_obj = getattr(obj, "__self__", None)
    if self_obj is None:
        return (), ("unbound", obj)

    func = getattr(obj, "__func__", None)
    if func is not None:
        # `types.MethodType`: can be reconstructed via `types.MethodType(func, self_obj)`.
        return (self_obj,), ("func", func, obj)

    name = getattr(obj, "__name__", None)
    if isinstance(name, str):
        # `types.BuiltinMethodType` / `types.MethodWrapperType`: try `getattr(self_obj, name)`.
        return (self_obj,), ("name", name, obj)

    return (self_obj,), ("obj", obj)


def _unflatten_bound_method(ctx: _BoundMethodCtx, children: tuple[Any, ...]) -> Any:
    if len(children) == 0:
        return ctx[1] if ctx[0] == "unbound" else ctx[-1]

    if len(children) != 1:
        return ctx[1] if ctx[0] == "unbound" else ctx[-1]

    (self_obj,) = children

    if ctx[0] == "func":
        func = ctx[1]
        fallback = ctx[2]
        try:
            return types.MethodType(func, self_obj)
        except TypeError:
            get = getattr(func, "__get__", None)
            if get is not None:
                try:
                    return get(self_obj, type(self_obj))
                except Exception:
                    pass
            return fallback

    if ctx[0] == "name":
        name = ctx[1]
        fallback = ctx[2]
        try:
            return getattr(self_obj, name)
        except Exception:
            return fallback

    return ctx[1] if ctx[0] in ("obj", "unbound") else ctx[-1]


def _register_bound_method_nodes() -> None:
    # Register Python's bound method objects as pytrees, so call-wrapping can treat
    # the bound `self` as part of the input tree for alias / mutation checks.
    #
    # This registration is namespaced to this module via `pytree = reexport(...)`.
    for cls in (types.MethodType, types.BuiltinMethodType, types.MethodWrapperType):
        try:
            cast(Any, pytree).register_node(
                cls, _flatten_bound_method, _unflatten_bound_method, namespace=__name__
            )
        except ValueError:
            # Already registered for this namespace.
            pass


_register_bound_method_nodes()


def tree_flatten[T](tree: PyTree[T]) -> tuple[list[T], pytree.PyTreeSpec]:
    return pytree.flatten(cast(optree.PyTree[T], tree))


@dataclass(frozen=True, kw_only=True)
class ValSpec[T: VarT = VarT]:
    type: type[T]
    mutable: bool

    def __repr__(self) -> str:
        if not self.mutable:
            return self.type.__name__
        return f"{self.type.__name__}:mut"

    @staticmethod
    def from_uservalue[T1: VarT](v: UserValue[T1]) -> ValSpec[T1]:
        if isinstance(v, VarT):
            return ValSpec(type=type(v), mutable=False)
        if not isinstance(v, VarRead):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(  # pyright: ignore[reportUnreachable]
                f"unsupported type: {type(v)} (value: {v})"
            )
        return ValSpec(type=v._get_type(), mutable=isinstance(v, Variable) and not v._read_only)

    def can_cast_to(self, other: ValSpec[T], /, *, check_mut: bool) -> bool:
        if not can_cast_implicit(self.type, other.type):
            return False
        if check_mut and not self.mutable and other.mutable:
            return False
        return True

    def promote(self, other: ValSpec):
        return ValSpec(type=promote_types(self.type, other.type), mutable=False)


@dataclass
class _OffsetProxy:
    offset: int
    spec: ValSpec

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.offset
        yield self.spec


@dataclass(frozen=True)
class TreeSpec[T = Any]:
    tree: pytree.PyTreeSpec
    leaves: list[ValSpec]

    @property
    def types(self) -> VarTS:
        return tuple(x.type for x in self.leaves)

    def as_schema(self) -> T:
        return self.tree.unflatten(self.leaves)  # pyright: ignore[reportReturnType]

    @staticmethod
    def from_schema[T1](val: T1) -> TreeSpec[T1]:
        leaves, tree = pytree.flatten(cast_unchecked(val))

        out: list[ValSpec] = []
        for leaf in leaves:
            if isinstance(leaf, ValSpec):
                out.append(cast(ValSpec, leaf))
            elif isinstance(leaf, type):
                if not issubclass(leaf, VarT):
                    raise TypeError(f"unsupported type: {leaf}")
                out.append(ValSpec(type=leaf, mutable=False))
            else:
                out.append(ValSpec.from_uservalue(leaf))

        return TreeSpec(tree, out)

    @staticmethod
    def primitive[T1: VarT](typ: type[T1], mutable: bool = False) -> TreeSpec[UserValue[T1]]:
        assert issubclass(typ, VarT)
        return TreeSpec(optree.treespec.leaf(), [ValSpec(type=typ, mutable=mutable)])

    @staticmethod
    def flatten[T1](val: T1) -> tuple[list[UserValue], TreeSpec[T1]]:
        leaves, tree = pytree.flatten(cast_unchecked(val))
        return leaves, TreeSpec(tree, [ValSpec.from_uservalue(x) for x in leaves])

    def can_cast_to(self, other: TreeSpec[T], /, *, check_mut: bool) -> bool:
        if self.tree != other.tree:
            return False
        for x, y in zip(self.leaves, other.leaves):
            if not x.can_cast_to(y, check_mut=check_mut):
                return False
        return True

    def can_cast_to_or_error(self, other: TreeSpec[T], /, *, check_mut: bool) -> None:
        if not self.can_cast_to(other, check_mut=check_mut):
            raise TypeError(f"expected {other}, got {self}")

    def flatten_up_to(self, val: T, /, *, check_mut: bool) -> list[UserValue]:
        """flatten val and check that val can be cast to self wrt types, but allows all mutability"""
        leaves, tree = TreeSpec.flatten(val)
        tree.can_cast_to_or_error(self, check_mut=check_mut)
        return leaves

    def unflatten_unchecked(self, leaves: Iterable[Any], /) -> T:
        return cast_unchecked(self.tree.unflatten(leaves))

    def unflatten(self, leaves: Iterable[UserValue], /, *, check_mut: bool) -> T:
        ans = self.unflatten_unchecked(leaves)
        _, spec = TreeSpec.flatten(ans)
        spec.can_cast_to_or_error(self, check_mut=check_mut)
        return ans

    def _unflatten_vals_ro(self, leaves: Iterable[Value], /) -> T:
        return self.unflatten(
            [Variable._from_val_ro(x, spec.type) for x, spec in zip(leaves, self.leaves)],
            check_mut=False,
        )

    def _unflatten_mvars(self, leaves: Iterable[MVar], /) -> T:
        return self.unflatten(
            [
                Variable(spec.type, _mvar=mv, _read_only=not spec.mutable)
                for mv, spec in zip(leaves, self.leaves)
            ],
            check_mut=True,
        )

    def __repr__(self) -> str:
        return repr(self.as_schema())

    def __len__(self) -> int:
        return len(self.leaves)

    def promote_types(self, other: TreeSpec[T]) -> TreeSpec[T]:
        if self.tree != other.tree:
            raise TypeError(f"incompatible types: {self} and {other}")
        out_types = [ValSpec.promote(t1, t2) for t1, t2 in zip(self.leaves, other.leaves)]
        return TreeSpec(self.tree, out_types)

    @staticmethod
    def promote_types_many[T1](*trees: TreeSpec[T1]) -> TreeSpec[T1]:
        x = trees[0]
        for tree in trees[1:]:
            x = x.promote_types(tree)
        return x

    def static_project[R](self, field: Callable[[T], R], /) -> R:
        proxy = self.unflatten_unchecked(None for _ in self.leaves)
        return field(proxy)

    def project[R](self, field: Callable[[T], R], /) -> tuple[list[int], TreeSpec[R]]:
        proxy = self.unflatten_unchecked(_OffsetProxy(i, typ) for i, typ in enumerate(self.leaves))
        res = field(proxy)

        leaves, out_tree = pytree.flatten(cast_unchecked(res))
        for l in leaves:
            if not isinstance(l, _OffsetProxy):
                raise TypeError(type(l), l)  # pyright: ignore[reportUnknownArgumentType]
        leaves = cast(list[_OffsetProxy], leaves)

        return [x.offset for x in leaves], TreeSpec(out_tree, [x.spec for x in leaves])

    def offset_of[R](self, field: Callable[[T], R], /) -> tuple[int, TreeSpec[R]]:
        idxs, subtree = self.project(field)

        if len(idxs) == 0:
            raise ValueError("result is empty, thus have no address")

        for x, y in zip(idxs[:-1], idxs[1:]):
            if y != x + 1:
                raise ValueError("not a continuous segment")

        return idxs[0], subtree


def copy_tree[T](v: T) -> T:
    vals, tree = TreeSpec.flatten(v)
    return tree._unflatten_vals_ro(read_uservalue(x) for x in vals)
