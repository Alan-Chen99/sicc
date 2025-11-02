from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Iterator
from typing import Literal
from typing import Protocol
from typing import cast
from typing import overload

import rich.repr
from rich.pretty import pretty_repr

from ._api import Constant
from ._api import Int
from ._api import UserValue
from ._api import Variable
from ._api import read_uservalue
from ._api import undef
from ._control_flow import break_
from ._control_flow import if_
from ._control_flow import loop
from ._control_flow import wrap_iterator_fn
from ._core import BoundInstr
from ._core import StaticBuffer
from ._core import Var
from ._core import VarT
from ._diagnostic import must_use
from ._instructions import ReadStack
from ._instructions import StackOpChain
from ._instructions import WriteStack
from ._stationeers import Pin
from ._tree_utils import TreeSpec
from ._tree_utils import field as optree_field  # pyright: ignore[reportUnknownVariableType]
from ._tree_utils import optree_dataclass
from ._utils import cast_unchecked
from ._utils import get_id


class PointerProto[T_co](Protocol):
    def _typing_helper(self) -> T_co: ...


@optree_dataclass(kw_only=True)
class Pointer[T = Any]:
    _device: Pin = Pin.db()
    _addr: Int
    _tree: TreeSpec[T] = optree_field(pytree_node=False)

    def _typing_helper(self) -> T: ...

    def __rich_repr__(self) -> rich.repr.Result:
        from ._stationeers import LiteralPin

        if isinstance((idx := self._device._idx), LiteralPin) and idx == LiteralPin("db"):
            pass
        else:
            yield self._device
        yield self._addr
        yield self._tree

    def __repr__(self):
        return pretty_repr(self)

    @staticmethod
    def _from_raw[T1](addr: Int, schema: T1, device: Pin = Pin.db()) -> Pointer[T1]:
        return Pointer(_device=device, _addr=addr, _tree=TreeSpec.from_schema(schema))

    def read(self) -> T:
        pin = read_uservalue(self._device._pin())
        addr = read_uservalue(self._addr)

        with must_use():
            out_vars: list[Var] = []
            instrs: list[BoundInstr[ReadStack]] = []

            for offset, typ in enumerate(self._tree.types):
                (out_v,), instr = ReadStack(typ, offset).create_bind(pin, addr)
                out_vars.append(out_v)
                instrs.append(instr)

            StackOpChain.from_parts(*instrs).emit()

        return self._tree._unflatten_vals_ro(out_vars)

    @overload
    def write[T1: VarT](self: Pointer[Variable[T1]], v: UserValue[T1], /) -> None: ...
    @overload
    def write(self, v: T, /) -> None: ...

    def write(self, v: T | Any, /) -> None:
        pin = read_uservalue(self._device._pin())
        addr = read_uservalue(self._addr)
        v_flat = [read_uservalue(x) for x in self._tree.flatten_up_to(v)]

        with must_use():
            instrs: list[BoundInstr[WriteStack]] = []
            for offset, (arg, typ) in enumerate(zip(v_flat, self._tree.types)):
                (), instr = WriteStack(typ, offset).create_bind(pin, addr, arg)
                instrs.append(instr)

            StackOpChain.from_parts(*instrs).emit()

    def _check_list[E](self: Pointer[list[E]]) -> tuple[TreeSpec[E], int]:
        schema = self._tree.as_schema()

        if not isinstance(schema, list):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"only lists are supported; got {schema}")

        items = [TreeSpec.from_schema(x) for x in schema]
        elem_tree = TreeSpec.promote_types_many(*items)

        return elem_tree, len(items)

    def _getitem_impl(self, idx: Int, /) -> Pointer:
        if isinstance(idx, int):
            return self.project(lambda x: cast_unchecked(x)[idx])

        elem_tree, _ = cast(Pointer[list[Any]], self)._check_list()
        return Pointer(
            _device=self._device,
            _addr=self._addr + len(elem_tree) * idx,
            _tree=elem_tree,
        )

    # @property
    # def value(self) -> T:
    #     return self.read()

    # @value.setter
    # def value(self, val: T) -> None:
    #     self.write(val)

    def static_project[R](self, field: Callable[[T], R]) -> R:
        return self._tree.static_project(field)

    def offset_of(self, field: Callable[[T], Any], /) -> int:
        ans, _ = self._tree.offset_of(field)
        return ans

    def project[S](self, field: Callable[[T], S], /) -> Pointer[S]:
        offset, out_tree = self._tree.offset_of(field)
        return Pointer(
            _device=self._device,
            _addr=self._addr + offset,
            _tree=out_tree,
        )

    # tuple overloads
    @overload
    def __getitem__[E](
        self: PointerProto[tuple[E, *tuple[Any, ...]]], idx: Literal[0], /
    ) -> Pointer[E]: ...
    @overload
    def __getitem__[E](
        self: PointerProto[tuple[Any, E, *tuple[Any, ...]]], idx: Literal[1], /
    ) -> Pointer[E]: ...
    @overload
    def __getitem__[E](
        self: PointerProto[tuple[Any, Any, E, *tuple[Any, ...]]], idx: Literal[2], /
    ) -> Pointer[E]: ...

    # other sequences
    @overload
    def __getitem__[E](self: PointerProto[SupportsGetItem[int, E]], idx: int, /) -> Pointer[E]: ...

    # dynamic index
    @overload
    def __getitem__[E](self: PointerProto[list[E]], idx: Int, /) -> Pointer[E]: ...

    def __getitem__(self, idx: Int, /) -> Pointer:
        return self._getitem_impl(idx)

    @overload
    def __setitem__[E](
        self: PointerProto[tuple[E, *tuple[Any, ...]]], idx: Literal[0], val: E, /
    ) -> None: ...
    @overload
    def __setitem__[E](
        self: PointerProto[tuple[Any, E, *tuple[Any, ...]]], idx: Literal[1], val: E, /
    ) -> None: ...
    @overload
    def __setitem__[E](
        self: PointerProto[tuple[Any, Any, E, *tuple[Any, ...]]], idx: Literal[2], val: E, /
    ) -> None: ...

    # other sequences
    @overload
    def __setitem__[E](
        self: PointerProto[SupportsGetItem[int, E]], idx: int, val: E, /
    ) -> None: ...

    # dynamic index
    @overload
    def __setitem__[E](self: PointerProto[list[E]], idx: Int, val: E, /) -> None: ...

    def __setitem__(self, idx: Int, val: Any, /) -> None:
        self._getitem_impl(idx).write(val)

    @wrap_iterator_fn
    def iter_mut[E](self: Pointer[list[E]]) -> Iterator[Pointer[E]]:
        elem_tree, length = self._check_list()
        assert length != 0

        size = len(elem_tree)

        # iterator points to the end of the object
        # so that object can be read directly by
        # pop %x

        it = Variable(self._addr + size)
        end = self._addr + (size + size * length)

        with loop():
            yield Pointer(
                _device=self._device,
                _addr=it - size,
                _tree=elem_tree,
            )
            it.value += size

            with if_(it == end):
                break_()

    @wrap_iterator_fn
    def iter_mut_rev[E](self: Pointer[list[E]]) -> Iterator[Pointer[E]]:
        elem_tree, length = self._check_list()
        assert length != 0
        size = len(elem_tree)

        # end ptr of self
        it = Variable(self._addr + size * length)

        with loop():
            yield Pointer(
                _device=self._device,
                _addr=it - size,
                _tree=elem_tree,
            )
            it.value -= size

            with if_(it == self._addr):
                break_()

    def __iter__[E](self: Pointer[list[E]]) -> Iterator[E]:
        for x in self.iter_mut():
            yield x.read()

    def iter_rev[E](self: Pointer[list[E]]) -> Iterator[E]:
        for x in self.iter_mut_rev():
            yield x.read()


class SupportsGetItem[K, V](Protocol):
    def __getitem__(self, key: K, /) -> V: ...


@overload
def stack_var[T: VarT](typ: type[T], /, device: Pin | None = None) -> Pointer[UserValue[T]]: ...
@overload
def stack_var[T: VarT](
    init: UserValue[T], /, device: Pin | None = None
) -> Pointer[UserValue[T]]: ...
@overload
def stack_var[T](init: T, /, device: Pin | None = None) -> Pointer[T]: ...


def stack_var[T](init_or_typ: T | Any, /, device: Pin | None = None) -> Pointer[T | Any]:
    """
    make a (global) variable on the stack, and return a pointer to it
    """
    from ._stationeers import Pin

    tree = TreeSpec.from_schema(init_or_typ)
    leaves: list[Any] = tree.tree.flatten_up_to(cast_unchecked(init_or_typ))

    ans = Pointer(
        _device=device or Pin.db(),
        _addr=Constant(StaticBuffer(get_id(), tree.types)),
        _tree=tree,
    )

    if all(isinstance(l, type) for l in leaves):
        return ans
    leaves = [undef(l) if isinstance(l, type) else l for l in leaves]
    ans.write(tree.unflatten(leaves))
    return ans


################################################################################
