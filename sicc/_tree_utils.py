from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import cast

import optree.treespec
import rich.repr
from optree.pytree import reexport
from rich import print as print  # autoflake: skip

from ._api import UserValue
from ._api import Variable
from ._api import _get_type
from ._api import read_uservalue
from ._core import Value
from ._core import VarT
from ._core import VarTS
from ._core import can_cast_implicit_many
from ._core import can_cast_implicit_many_or_err
from ._core import promote_types
from ._utils import ReprAs
from ._utils import cast_unchecked

if TYPE_CHECKING:
    from dataclasses import dataclass as _optree_dataclass

    optree_dataclass = _optree_dataclass

    from optree import dataclasses as dataclasses
    from optree import pytree as pytree

    field = dataclasses.field  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
else:
    pytree = reexport(namespace=__name__)
    dataclasses = pytree.dataclasses
    optree_dataclass = dataclasses.dataclass
    field = dataclasses.field


@dataclass
class _OffsetProxy:
    offset: int
    typ: type[VarT]

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.offset
        yield ReprAs(self.typ.__qualname__)


@dataclass
class TreeSpec[T = Any]:
    tree: pytree.PyTreeSpec
    types: VarTS

    def as_schema(self) -> T:
        return self.unflatten_unchecked(self.types)

    @staticmethod
    def from_schema[T1](val: T1) -> TreeSpec[T1]:
        leaves, tree = pytree.flatten(cast_unchecked(val))

        def handle_leaf(leaf: Any) -> type[VarT]:
            if not isinstance(leaf, type):
                return _get_type(leaf)
            if not issubclass(leaf, VarT):
                raise TypeError(f"unsupported type: {leaf}")
            return leaf

        return TreeSpec(tree, tuple(handle_leaf(l) for l in leaves))

    @staticmethod
    def primitive[T1: VarT](typ: type[T1]) -> TreeSpec[UserValue[T1]]:
        return TreeSpec(optree.treespec.leaf(), (typ,))

    @staticmethod
    def flatten[T1](val: T1) -> tuple[list[UserValue], TreeSpec[T1]]:
        leaves, tree = pytree.flatten(cast_unchecked(val))
        return leaves, TreeSpec(tree, tuple(_get_type(x) for x in leaves))

    def can_cast_implicit_many_or_err(self, other: TreeSpec[T]) -> None:
        if not can_cast_implicit_many(self.types, other.types):
            raise TypeError(f"not possible to use {self} as {other}")

    def flatten_up_to(self, val: T) -> list[UserValue]:
        leaves, tree = TreeSpec.flatten(val)
        tree.can_cast_implicit_many_or_err(self)
        return leaves

    def unflatten_unchecked(self, leaves: Iterable[Any], /) -> T:
        return cast_unchecked(self.tree.unflatten(leaves))

    def unflatten(self, leaves_: Iterable[UserValue], /) -> T:
        leaves = list(leaves_)
        can_cast_implicit_many_or_err(self.types, tuple(_get_type(x) for x in leaves))
        return self.unflatten_unchecked(leaves)

    def _unflatten_vals_ro(self, leaves: Iterable[Value], /) -> T:
        return self.unflatten([Variable._from_val_ro(x, typ) for x, typ in zip(leaves, self.types)])

    def __repr__(self) -> str:
        return repr(self.tree.unflatten(ReprAs(x.__name__) for x in self.types))

    def __len__(self) -> int:
        return len(self.types)

    def promote_types(self, other: TreeSpec[T]) -> TreeSpec[T]:
        if self.tree != other.tree:
            raise TypeError(f"incompatible types: {self} and {other}")
        out_types = tuple(promote_types(t1, t2) for t1, t2 in zip(self.types, other.types))
        return TreeSpec(self.tree, out_types)

    @staticmethod
    def promote_types_many[T1](*trees: TreeSpec[T1]) -> TreeSpec[T1]:
        x = trees[0]
        for tree in trees[1:]:
            x = x.promote_types(tree)
        return x

    def static_project[R](self, field: Callable[[T], R], /) -> R:
        proxy = self.unflatten_unchecked(None for _ in self.types)
        return field(proxy)

    def project[R](self, field: Callable[[T], R], /) -> tuple[list[int], TreeSpec[R]]:
        proxy = self.unflatten_unchecked(_OffsetProxy(i, typ) for i, typ in enumerate(self.types))
        res = field(proxy)

        leaves, out_tree = pytree.flatten(cast_unchecked(res))
        for l in leaves:
            if not isinstance(l, _OffsetProxy):
                raise TypeError(type(l), l)  # pyright: ignore[reportUnknownArgumentType]
        leaves = cast(list[_OffsetProxy], leaves)

        return [x.offset for x in leaves], TreeSpec(out_tree, tuple(x.typ for x in leaves))

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
