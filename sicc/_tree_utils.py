from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
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
from ._core import Value
from ._core import VarT
from ._core import VarTS
from ._core import can_cast_implicit
from ._core import promote_types
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

type PyTree[T] = T | Sequence[PyTree[T]] | Mapping[Any, PyTree[T]] | optree.PyTree[T]


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

    def can_cast_to(self, other: ValSpec[T]) -> bool:
        if not can_cast_implicit(self.type, other.type):
            return False
        if not self.mutable and other.mutable:
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

    def can_cast_to(self, other: TreeSpec[T]) -> bool:
        if self.tree != other.tree:
            return False
        for x, y in zip(self.leaves, other.leaves):
            if not x.can_cast_to(y):
                return False
        return True

    def can_cast_to_or_error(self, other: TreeSpec[T]) -> None:
        if not self.can_cast_to(other):
            raise TypeError(f"not possible to use {self} as {other}")

    def flatten_up_to(self, val: T) -> list[UserValue]:
        leaves, tree = TreeSpec.flatten(val)
        tree.can_cast_to_or_error(self)
        return leaves

    def unflatten_unchecked(self, leaves: Iterable[Any], /) -> T:
        return cast_unchecked(self.tree.unflatten(leaves))

    def unflatten(self, leaves: Iterable[UserValue], /) -> T:
        ans = self.unflatten_unchecked(leaves)
        _, spec = TreeSpec.flatten(ans)
        spec.can_cast_to_or_error(self)
        return ans

    def _unflatten_vals_ro(self, leaves: Iterable[Value], /) -> T:
        return self.unflatten(
            [Variable._from_val_ro(x, spec.type) for x, spec in zip(leaves, self.leaves)]
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
