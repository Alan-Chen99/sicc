from __future__ import annotations

import abc
import functools
import inspect
import random
import string
from contextlib import AbstractContextManager
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Concatenate
from typing import Final
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Never
from typing import Protocol
from typing import Self
from typing import Sequence
from typing import TypedDict
from typing import TypeVar
from typing import Unpack
from typing import cast
from typing import overload
from typing import override

import rich.repr
from rich.pretty import pretty_repr
from rich.text import Text

from . import _functions as _f
from ._core import AnyType
from ._core import AsRawCtx
from ._core import BoundInstr
from ._core import EffectBase
from ._core import InstrBase
from ._core import InstrTypedWithArgs_api
from ._core import Label
from ._core import LabelLike
from ._core import MVar
from ._core import RawText
from ._core import Scope
from ._core import StaticBuffer
from ._core import TypeList
from ._core import Undef
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VarTS
from ._core import VirtualConst
from ._core import can_cast_implicit_many
from ._core import can_cast_implicit_many_or_err
from ._core import format_raw_val
from ._core import format_val
from ._core import get_type
from ._core import get_types
from ._core import nan
from ._core import promote_types
from ._diagnostic import DebugInfo
from ._diagnostic import Warnings
from ._diagnostic import add_debug_info
from ._diagnostic import debug_info
from ._diagnostic import describe_fn
from ._diagnostic import must_use
from ._diagnostic import register_exclusion
from ._diagnostic import suppress_warnings
from ._diagnostic import track_caller
from ._instructions import XORB
from ._instructions import XORI
from ._instructions import AbsF
from ._instructions import AbsI
from ._instructions import AddF
from ._instructions import AddI
from ._instructions import AndB
from ._instructions import AndI
from ._instructions import AsmBlock
from ._instructions import BlackBox
from ._instructions import DivF
from ._instructions import EffectExternal
from ._instructions import Jump
from ._instructions import LShift
from ._instructions import Max
from ._instructions import Min
from ._instructions import MulF
from ._instructions import MulI
from ._instructions import Not
from ._instructions import OrB
from ._instructions import OrI
from ._instructions import PredEq
from ._instructions import PredLE
from ._instructions import PredLT
from ._instructions import PredNEq
from ._instructions import RawInstr
from ._instructions import ReadStack
from ._instructions import RShiftSigned
from ._instructions import RShiftUnsigned
from ._instructions import Select
from ._instructions import StackOpChain
from ._instructions import SubF
from ._instructions import SubI
from ._instructions import Transmute
from ._instructions import UnreachableChecked
from ._instructions import WriteStack
from ._tracing import _CUR_SCOPE
from ._tracing import RawSubr
from ._tracing import break_
from ._tracing import ensure_label
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import mk_mvar
from ._tracing import trace_if
from ._tracing import trace_to_raw_subr
from ._tracing import trace_while
from ._tree_utils import dataclass as optree_dataclass
from ._tree_utils import field as optree_field  # pyright: ignore[reportUnknownVariableType]
from ._tree_utils import pytree
from ._utils import Cell
from ._utils import ReprAs
from ._utils import cast_unchecked
from ._utils import empty
from ._utils import empty_t
from ._utils import get_id
from ._utils import isinst
from ._utils import late_fn

if TYPE_CHECKING:
    from ._stationeers import Pin

register_exclusion(__file__)

T_co = TypeVar("T_co", covariant=True, bound=VarT, default=Any)

type UserValue[T: VarT = VarT] = VarRead[T] | T

Bool = UserValue[bool]
Int = UserValue[int]
Str = UserValue[str]

# TODO: this causes pyright to report
# Type of parameter "x" is unknown (reportUnknownParameterType)
# on function "greater_than"
# when using --threads, sometimes
# type Float = UserValue[float]
type Float = VarRead[float] | float

type ValLabelLike = UserValue[Label] | str


class VarRead[T: VarT](abc.ABC):
    """
    covariant readonly base
    """

    # TODO: its possible to move Function up before VarRead and
    # then get rid of all the late_fn here?

    @abc.abstractmethod
    def _read(self) -> Value[T]: ...

    @abc.abstractmethod
    def _get_type(self) -> type[T]: ...

    def __check_co(self) -> VarRead[VarT]:  # pyright: ignore[reportUnusedFunction]
        return self

    def __bool__(self) -> Never:
        err = TypeError("not possible to convert a runtime value to a compile-time bool.")
        err.add_note("use tilde operator '~' to invert a boolean")
        raise err

    # explicit @overload works with protocol requiring __add__ (so sum(), etc)
    # maybe write explicit @overload for all?
    @overload
    def __add__(self: Int, other: Int, /) -> VarRead[int]: ...
    @overload
    def __add__(self: Float, other: Float, /) -> VarRead[float]: ...
    def __add__(self: Float, other: Float, /) -> VarRead[float]:
        return add(self, other)

    __radd__ = __add__

    __sub__ = late_fn(lambda: sub)
    __rsub__ = late_fn(lambda: sub._rev_same_sig())

    __mul__ = late_fn(lambda: mul)
    __rmul__ = late_fn(lambda: mul)

    __truediv__ = late_fn(lambda: div)
    __rtruediv__ = late_fn(lambda: div._rev_same_sig())

    __invert__ = late_fn(lambda: not_)

    __and__ = late_fn(lambda: and_)
    __rand__ = late_fn(lambda: and_)
    __or__ = late_fn(lambda: or_)
    __ror__ = late_fn(lambda: or_)
    __xor__ = late_fn(lambda: xor)
    __rxor__ = late_fn(lambda: xor)

    __lshift__ = late_fn(lambda: lshift)
    __rshift__ = late_fn(lambda: rshift_signed)

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Any
    ) -> VarRead[bool]:
        return equal(self, other)

    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Any
    ) -> VarRead[bool]:
        return not_equal(self, other)

    __lt__ = late_fn(lambda: less_than)
    __le__ = late_fn(lambda: less_than_or_eq)

    __gt__ = late_fn(lambda: greater_than)
    __ge__ = late_fn(lambda: greater_than_or_eq)

    def __abs__(self: VarRead[float]) -> VarRead[float]:
        return abs_.call(self)

    def is_nan(self) -> VarRead[bool]:
        return equal(self, nan)

    def transmute[O: VarT](self, out_type: type[O] = AnyType) -> VarRead[O]:
        return transmute(self, out_type)


def _get[T: VarT](v: UserValue[T]) -> Value[T]:
    if isinstance(v, VarT):
        return v
    return v._read()


def _get_label(l: ValLabelLike) -> Value[Label]:
    if isinstance(l, str):
        return ensure_label(l)
    return _get(l)


def label_ref(l: str | None = None, *, unique: bool = False) -> Label:
    """make a label to be emitted later"""
    return ensure_label(l)


def _get_type[T: VarT](v: UserValue[T]) -> type[T]:
    if isinstance(v, VarT):
        return type(v)
    if not isinstance(v, VarRead):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"unsupported type: {type(v)} (value: {v})"
        )
    return v._get_type()


class VariableOpts(TypedDict, total=False):
    _read_only: bool
    _mvar: MVar | None


class Variable[T: VarT](VarRead[T]):
    """
    a MVar exposed to the user.

    This class is needed because Var/MVar have equality and compareison by id
    (needed in compiler internals); Variable equality will instead stage a equality operation.
    """

    _inner: MVar[T]
    _read_only: bool

    @overload
    def __init__(self, typ: type[T], /, **kwargs: Unpack[VariableOpts]) -> None: ...
    @overload
    def __init__(self, init: UserValue[T], /, **kwargs: Unpack[VariableOpts]) -> None: ...
    @overload
    def __init__(
        self, typ: type[T], init: UserValue[T], /, **kwargs: Unpack[VariableOpts]
    ) -> None: ...
    def __init__(
        self,
        init_or_typ: type[T] | UserValue[T],
        init: UserValue[T] | None = None,
        /,
        **kwargs: Unpack[VariableOpts],
    ) -> None:
        read_only = kwargs.get("_read_only", False)
        mvar = kwargs.get("_mvar", None)

        self._read_only = read_only

        if mvar:
            assert mvar.type == init_or_typ
            self._inner = mvar
            return

        if init is not None:
            assert isinstance(init_or_typ, type)
            self._inner = mk_mvar(init_or_typ)
            self._inner.write(_get(init))
            return

        if isinstance(init_or_typ, type):
            self._inner = mk_mvar(init_or_typ)
            return

        x_val = _get(init_or_typ)
        typ = get_type(x_val)
        if not read_only and typ == AnyType:
            raise TypeError("cannot infer type; specify one explicitly")
        self._inner = mk_mvar(typ)
        self._inner.write(x_val)

    def __repr__(self) -> str:
        return repr(self._inner)

    @staticmethod
    def _from_val_ro[T1: VarT](v: Value[T1], typ: type[T1] | None = None) -> VarRead[T1]:
        ans = Variable(typ or get_type(v), _read_only=True)
        ans._inner.write(v)
        return ans

    @override
    def _read(self) -> Value[T]:
        return self._inner.read()

    @override
    def _get_type(self) -> type[T]:
        return self._inner.type

    @property
    def value(self) -> VarRead[T]:
        return Variable(self, _read_only=True)

    @value.setter
    def value(self, v: UserValue[T]) -> None:
        if self._read_only:
            raise TypeError(f"writing read-only variable {self}")
        self._inner.write(_get(v))

    @staticmethod
    def undef[T1: VarT](typ: type[T1]) -> Variable[T1]:
        return Variable(undef(typ))


@dataclass(frozen=True)
class _Constant[T: VarT](VarRead[T]):
    val: VirtualConst[T]

    def __repr__(self):
        return repr(self.val)

    @override
    def _get_type(self) -> type[T]:
        return self.val.get_type()

    @override
    def _read(self) -> Value[T]:
        return self.val


class _FunctionProto[T_co]:
    def _instrs_type_helper(self) -> T_co: ...


class Function[*Ts](_FunctionProto[tuple[*Ts]]):
    """
    a possibly overloaded function that maps to a instruction
    """

    _instrs: tuple[InstrBase, ...]

    def __init__(self, *overloads: *Ts) -> None:
        for x in overloads:
            assert isinstance(x, InstrBase)
        self._instrs = cast_unchecked(overloads)

    def _instrs_type_helper(self) -> tuple[*Ts]:
        return cast_unchecked(self._instrs)

    def __repr__(self) -> str:
        return "Function(" + ", ".join(type(x).__name__ for x in self._instrs) + ")"

    @overload
    def __get__(self, obj: None, objtype: type) -> Self: ...
    @overload
    def __get__[V](self, obj: V, objtype: Any) -> _BoundFunction[V, tuple[*Ts]]: ...

    def __get__(self, obj: Any, objtype: Any) -> Any:
        """
        typing helper to get type checker accept Function as a method
        for methods to actually work one must use for ex late_fn
        """
        assert False

    @overload
    def __call__[*I, O](
        self: _FunctionProto[tuple[InstrTypedWithArgs_api[tuple[*I], O], *tuple[Any, ...]]],
        *args: *I,
    ) -> O: ...
    @overload
    def __call__[*I, O](
        self: _FunctionProto[tuple[Any, InstrTypedWithArgs_api[tuple[*I], O], *tuple[Any, ...]]],
        *args: *I,
    ) -> O: ...

    def __call__(self, *args: Any) -> Any:
        arg_types = tuple(_get_type(x) for x in args)

        for instr in self._instrs:
            assert isinstance(instr, InstrBase)
            if not can_cast_implicit_many(arg_types, instr.in_types):
                continue

            argvals = tuple(_get(x) for x in args)
            with must_use(), track_caller():
                ans = cast_unchecked(instr.call(*argvals))  # pyright: ignore
            if ans is None:
                return None
            assert isinst(ans, Var)

            return Variable._from_val_ro(ans)

        can_cast_implicit_many_or_err(arg_types, self._instrs[-1].in_types)
        assert False

    call = __call__

    def _rev_same_sig(self) -> Self:
        def inner(a1: Any, a2: Any) -> Any:
            return self.call(a2, a1)  # pyright: ignore

        return cast_unchecked(inner)


class _BoundFunction[V, Ts: tuple[Any, ...]](Protocol):
    @overload
    def __call__[V1, *I, O](
        self: _BoundFunction[V1, tuple[InstrTypedWithArgs_api[tuple[V1, *I], O], *tuple[Any, ...]]],
        *args: *I,
    ) -> O: ...
    @overload
    def __call__[V1, *I, O](
        self: _BoundFunction[
            V1, tuple[Any, InstrTypedWithArgs_api[tuple[V1, *I], O], *tuple[Any, ...]]
        ],
        *args: *I,
    ) -> O: ...


################################################################################


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
    def flatten[T1](val: T1) -> tuple[list[UserValue], TreeSpec[T1]]:
        leaves, tree = pytree.flatten(cast_unchecked(val))
        return leaves, TreeSpec(tree, tuple(_get_type(x) for x in leaves))

    def flatten_up_to(self, val: T) -> list[UserValue]:
        return cast_unchecked(self.tree.flatten_up_to(cast_unchecked(val)))

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

    def project[R](self, field: Callable[[T], R], /) -> tuple[list[int], TreeSpec[R]]:
        proxy = self.unflatten_unchecked(_OffsetProxy(i, typ) for i, typ in enumerate(self.types))
        res = field(proxy)

        leaves, out_tree = pytree.flatten(cast_unchecked(res))
        for l in leaves:
            assert isinstance(l, _OffsetProxy)
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
    return tree._unflatten_vals_ro(_get(x) for x in vals)


################################################################################


@dataclass
class State[T = Any]:
    _scope: Scope | None
    _tree: TreeSpec[T] | None = None
    _vars: list[MVar] | None = None

    def __init__(self, init: T | empty_t = empty, *, _tree: TreeSpec[T] | None = None):
        self._scope = _CUR_SCOPE.get()
        if _tree:
            self._tree = _tree
            self._vars = [mk_mvar(t) for t in _tree.types]
        if not isinstance(init, empty_t):
            self.write(init)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self._tree

    def __repr__(self):
        return pretty_repr(self)

    def read(self) -> T:
        assert self._tree is not None
        assert self._vars is not None

        with track_caller():
            vars_ = [x.read() for x in self._vars]
        return self._tree._unflatten_vals_ro(vars_)

    def write(self, v: T):
        if self._tree is None:
            _leaves, tree = TreeSpec.flatten(v)
            assert self._vars is None
            self._tree = tree
            if self._scope:
                with _CUR_SCOPE.bind(self._scope):
                    self._vars = [mk_mvar(t) for t in tree.types]
            else:
                with _CUR_SCOPE.bind_clear():
                    self._vars = [mk_mvar(t) for t in tree.types]

        leaves = self._tree.flatten_up_to(v)
        vars = [_get(x) for x in leaves]

        assert self._vars is not None
        with track_caller():
            for mv, arg in zip(self._vars, vars):
                mv.write(arg)

    @property
    def value(self) -> T:
        return self.read()

    @value.setter
    def value(self, val: T) -> None:
        self.write(val)

    def ref_mut(self) -> T:
        assert self._tree is not None
        assert self._vars is not None

        return self._tree.unflatten(Variable(x.type, _mvar=x) for x in self._vars)

    def project[R](self, field: Callable[[T], R], /) -> State[R]:
        assert self._tree is not None
        assert self._vars is not None

        offsets, tree = self._tree.project(field)

        ans = State(_tree=tree)
        ans._vars = [self._vars[i] for i in offsets]
        return ans


################################################################################


@dataclass
class _OffsetProxy:
    offset: int
    typ: type[VarT]

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.offset
        yield ReprAs(self.typ.__qualname__)


class PointerProto[T_co](Protocol):
    def _typing_helper(self) -> T_co: ...


@optree_dataclass(kw_only=True)
class Pointer[T = Any]:
    _device: Pin
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

    def read(self) -> T:
        pin = _get(self._device._pin())
        addr = _get(self._addr)

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
        pin = _get(self._device._pin())
        addr = _get(self._addr)
        v_flat = [_get(x) for x in self._tree.flatten_up_to(v)]

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

    def __iter__[E](self: Pointer[list[E]]) -> Iterator[Pointer[E]]:
        elem_tree, length = self._check_list()
        assert length != 0

        idx = Variable(self._addr)
        end = self._addr + len(elem_tree) * length

        with loop():
            yield Pointer(
                _device=self._device,
                _addr=idx,
                _tree=elem_tree,
            )
            idx.value += len(elem_tree)

            with if_(idx == end):
                break_()


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
    leaves: list[Any] = tree.flatten_up_to(init_or_typ)

    ans = Pointer(
        _device=device or Pin.db(),
        _addr=_Constant(StaticBuffer(get_id(), tree.types)),
        _tree=tree,
    )

    if all(isinstance(l, type) for l in leaves):
        return ans
    leaves = [undef(l) if isinstance(l, type) else l for l in leaves]
    ans.write(tree.unflatten(leaves))
    return ans


################################################################################


class EnumEx(Enum):
    def as_raw(self, ctx: AsRawCtx) -> RawText:
        return RawText.str(repr(self.value))


################################################################################


@dataclass
class TracedSubr[F = Any]:
    subr: RawSubr
    arg_tree: TreeSpec
    ret_tree: TreeSpec

    @property
    def call[**P, R](self: TracedSubr[Callable[P, R]]) -> Callable[P, R]:
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            vals = self.arg_tree.flatten_up_to((args, kwargs))
            vals_ = tuple(_get(x) for x in vals)
            out_vars = self.subr.call(*vals_)

            out_vars_ = [Variable._from_val_ro(x) for x in out_vars]
            return self.ret_tree.unflatten(out_vars_)

        return inner


_RETURN_HOOK: Cell[Callable[[Any], None]] = Cell()


def return_(val: Any = None) -> None:
    return _RETURN_HOOK(val)


def trace_to_subr[**P, R](
    fn: Callable[P, R | None], *args: P.args, **kwargs: P.kwargs
) -> TracedSubr[Callable[P, R]]:
    arg_vars, arg_tree = TreeSpec.flatten((args, kwargs))
    arg_types = tuple(_get_type(x) for x in arg_vars)

    out_tree: Cell[TreeSpec[R]] = Cell()

    @functools.wraps(fn)
    def inner(*args: Var) -> tuple[Var, ...]:
        ar, kw = arg_tree.unflatten(Variable._from_val_ro(x) for x in args)

        id = get_id()

        subr_scope = _CUR_SCOPE.value
        exit_paths: list[tuple[Label, R, DebugInfo]] = []

        def ret_hook(val: Any) -> None:
            with _CUR_SCOPE.bind(subr_scope):
                exit_label = mk_internal_label(f"trace_to_subr_ret({len(exit_paths)})", id)
                # FIXME: this is a out of scope use;
                # we dont currenty error for that but will probably in future
                exit_paths.append((exit_label, copy_tree(val), debug_info()))
                jump(exit_label)

        with _RETURN_HOOK.bind(ret_hook):
            ans = fn(*ar, **kw)

            with add_debug_info(DebugInfo(describe=f"return val from end of {describe_fn(fn)}")):
                return_(ans)
                _f.unreachable_checked()

        ret_tree = TreeSpec.promote_types_many(
            *(TreeSpec.flatten(x)[1] for _, x, _ in exit_paths),
        )
        out_tree.value = ret_tree
        ret_state = State(_tree=ret_tree)

        actual_exit = mk_internal_label(f"trace_to_subr_exit", id)

        for exit_label, exit_val, debug in exit_paths:
            with add_debug_info(debug):
                label(exit_label)
                ret_state.write(exit_val)
                jump(actual_exit)

        label(actual_exit)

        assert ret_state._vars is not None
        return tuple(x.read() for x in ret_state._vars)

    ans = trace_to_raw_subr(arg_types, inner)

    return TracedSubr(ans, arg_tree, out_tree.value)


class Subr[F]:
    fn: Final[F]

    def __init__(self, fn: F) -> None:
        self.fn = fn
        self._subr: TracedSubr | None = None

    def __repr__(self):
        return repr(self.fn)

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
        bound = inspect.signature(self.fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        if self._subr is None:
            self._subr = trace_to_subr(
                self.fn, **bound.arguments
            )  # pyright: ignore[reportCallIssue]
        return self._subr.call(**bound.arguments)


@dataclass
class SubrFactory:
    inline_always: bool = False

    def __call__[F](self, fn: F) -> Subr[F]:
        if self.inline_always:
            raise NotImplementedError()
        return Subr(fn)


class SubrOpts(TypedDict, total=False):
    inline_always: bool


@overload
def subr(**kwargs: Unpack[SubrOpts]) -> SubrFactory: ...
@overload
def subr[F](func: F, /, **kwargs: Unpack[SubrOpts]) -> Subr[F]: ...


def subr[F](func: F | None = None, /, **kwargs: Unpack[SubrOpts]) -> SubrFactory | Subr[F]:
    config = SubrFactory(**kwargs)
    if func is None:
        return config
    return config(func)


@dataclass(kw_only=True)
class BlockRef[T]:
    _id: int
    _scope: Scope
    _break_paths: list[tuple[Label, T, DebugInfo]]
    _out_value: T | empty_t = empty

    exit_label: Label
    finished_tracing: bool

    def break_(self, val: T = None) -> None:
        with _CUR_SCOPE.bind(self._scope):
            break_label = mk_internal_label(f"block_break_({len(self._break_paths)})", self._id)
            # FIXME: this is a out of scope use;
            # we dont currenty error for that but will probably in future
            self._break_paths.append((break_label, copy_tree(val), debug_info()))
            jump(break_label)

    def get(self) -> T:
        if not self.finished_tracing:
            raise RuntimeError("can only get value after block ends")

        if len(self._break_paths) == 0:
            raise RuntimeError("not possible to get break value because break_ is never called")

        assert not isinstance(self._out_value, empty_t)
        return self._out_value

    @property
    def value(self) -> T:
        return self.get()


@overload
def block() -> AbstractContextManager[BlockRef[Any]]: ...
@overload
def block[T](_ret_typ: type[T], /) -> AbstractContextManager[BlockRef[T]]: ...


def block[T](_ret_typ: type[T] | None = None, /) -> AbstractContextManager[BlockRef[T]]:
    return block_impl(_ret_typ)


@contextmanager
def block_impl[T](_ret_typ: type[T] | None = None) -> Iterator[BlockRef[T]]:
    id = get_id()

    block_ref = BlockRef[T](
        _id=id,
        _scope=_CUR_SCOPE.value,
        _break_paths=[],
        exit_label=mk_internal_label(f"block_exit", id),
        finished_tracing=False,
    )

    yield block_ref

    block_ref.finished_tracing = True
    if len(block_ref._break_paths) == 0:
        return

    out_tree = TreeSpec.promote_types_many(
        *(TreeSpec.flatten(x)[1] for _, x, _ in block_ref._break_paths),
    )
    out_state = State(_tree=out_tree)

    actual_exit = mk_internal_label(f"block_exit", block_ref._id)

    for break_label, exit_val, debug in block_ref._break_paths:
        with add_debug_info(debug):
            label(break_label)
            out_state.write(exit_val)
            jump(actual_exit)

    label(actual_exit)

    ans = out_state.read()
    block_ref._out_value = ans


def inline_subr[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        with block() as b:

            def ret_hook(val: Any) -> None:
                b.break_(val)

            with _RETURN_HOOK.bind(ret_hook):
                ans = fn(*args, **kwargs)
                return_(ans)
                _f.unreachable_checked()

        return b.value

    return inner


################################################################################


def undef[T: VarT](typ: type[T] = AnyType) -> VarRead[T]:
    """return the undef constant"""
    return _Constant(Undef(typ))


################################################################################

add = Function(AddI(), AddF())
sub = Function(SubI(), SubF())
mul = Function(MulI(), MulF())
div = Function(DivF())

and_ = Function(AndB(), AndI())
or_ = Function(OrB(), OrI())
xor = Function(XORB(), XORI())

lshift = Function(LShift())
rshift_signed = Function(RShiftSigned())
rshift_unsigned = Function(RShiftUnsigned())

min_ = Function(Min(int), Min(float))
max_ = Function(Max(int), Max(float))


def clamp[T: float](
    val: UserValue[T],
    /,
    min: UserValue[T],
    max: UserValue[T],
) -> VarRead[T]:
    ans: VarRead[float] = min_(max_(val, min), max)
    return cast_unchecked(ans)


abs_ = Function(AbsI(), AbsF())

##

not_ = Function(Not())

less_than = Function(PredLT())
less_than_or_eq = Function(PredLE())


def greater_than(x: Float, y: Float) -> VarRead[bool]:
    return less_than(y, x)


def greater_than_or_eq(x: Float, y: Float) -> VarRead[bool]:
    return less_than_or_eq(y, x)


def equal[T: VarT](x: UserValue[T], y: UserValue[T]) -> VarRead[bool]:
    return Function(PredEq(promote_types(_get_type(x), _get_type(y)))).call(x, y)


def not_equal[T: VarT](x: UserValue[T], y: UserValue[T]) -> VarRead[bool]:
    return Function(PredNEq(promote_types(_get_type(x), _get_type(y)))).call(x, y)


##


def unreachable_checked() -> None:
    with suppress_warnings(Warnings.Unused):
        Function(UnreachableChecked()).call()


def black_box[T: VarT](v: UserValue[T]) -> VarRead[T]:
    return Function(BlackBox(_get_type(v))).call(v)


def transmute[O: VarT](v: UserValue, out_type: type[O] = AnyType) -> VarRead[O]:
    """
    currently often generate extra move instructions; might get fixed later
    """
    return Function(Transmute(_get_type(v), out_type)).call(v)


################################################################################


@dataclass(frozen=True)
class EffectComment(EffectBase):
    pass


@optree_dataclass
class _CommentStatic:
    text: Text = optree_field(pytree_node=False)


class Comment(InstrBase):
    def __init__(self, tree: TreeSpec[tuple[Any, ...]]) -> None:
        self.tree = tree
        self.in_types = cast(TypeList[tuple[Value, ...]], TypeList(tree.types))
        self.out_types = ()

    def format_with_vals(self, vals: list[Text], prefix: Text) -> Text:
        def random_seq(n: int):
            return "".join(random.choices(string.ascii_letters + string.digits, k=n))

        arg_placeholders = [random_seq(10) for _ in self.in_types]
        args = self.tree.unflatten_unchecked(ReprAs(x) for x in arg_placeholders)

        def handle_placeholders(s: str) -> Text:
            for x, text in zip(arg_placeholders, vals):
                before, sep, after = s.partition(x)
                if sep:
                    return Text() + handle_placeholders(before) + text + handle_placeholders(after)
            return Text(s, "ic10.comment")

        ans = Text()
        ans += prefix
        for arg in args:
            ans += " "
            if isinstance(arg, _CommentStatic):
                ans += arg.text
            else:
                ans += handle_placeholders(pretty_repr(arg, max_width=10000))

        return ans

    @override
    def format(self, instr: BoundInstr[Self]) -> Text:
        vals_text = [format_val(x, typ) for x, typ in zip(instr.inputs_, self.in_types)]
        return self.format_with_vals(vals_text, Text("*", "ic10.jump"))

    def format_raw(self, instr: BoundInstr[Self], ctx: AsRawCtx) -> RawText:
        ans = Text()
        args = [
            format_raw_val(x, ctx, t, instr.debug).text
            for t, x in zip(self.in_types, instr.inputs_)
        ]
        ans += self.format_with_vals(args, Text("#", "ic10.comment"))
        ans += "\n"
        return RawText(ans)

    # dont reorder comments
    reads_ = EffectComment()
    writes_ = EffectComment()


def comment(*args_: Any) -> None:
    args = tuple(
        _CommentStatic(Text(x, "ic10.comment")) if isinstance(x, str) else x for x in args_
    )
    vars, tree = TreeSpec.flatten(args)
    vars_ = [_get(v) for v in vars]
    return Comment(tree).call(*vars_)


################################################################################


def jump(label: ValLabelLike) -> None:
    return Jump().call(_get_label(label))


def branch(cond: Bool, on_true: ValLabelLike, on_false: ValLabelLike) -> None:
    return _f.branch(_get(cond), _get_label(on_true), _get_label(on_false))


def cjump(cond: Bool, on_true: ValLabelLike) -> None:
    cont = mk_internal_label("cjump_cont")
    _f.branch(_get(cond), _get_label(on_true), cont)
    label(cont)


def if_(cond: Bool) -> AbstractContextManager[None]:
    return trace_if(_get(cond))


def while_(cond_fn: Callable[[], Bool]) -> AbstractContextManager[None]:
    def inner():
        return _get(cond_fn())

    return trace_while(inner)


def loop() -> AbstractContextManager[None]:
    return trace_while(lambda: True)


@overload
def range_(stop: Int, /) -> Iterator[VarRead[int]]: ...
@overload
def range_(start: Int, stop: Int, /) -> Iterator[VarRead[int]]: ...


def range_(x: Int, y: Int | None = None, /) -> Iterator[VarRead[int]]:
    # will be implemented to support all args to python range in future
    if y is None:
        start, stop = 0, x
    else:
        start, stop = x, y

    if isinstance(start, int) and isinstance(stop, int) and start < stop:
        # do while is 1 less instruction
        idx = Variable(int, start)
        # some VarRead is defferred; make sure its only read once
        stop_ = Variable(int, stop)
        with loop():
            yield idx.value
            idx.value += 1
            with if_(idx.value == stop_):
                break_()
    else:
        idx = Variable(int, start)
        stop_ = Variable(int, stop)
        with while_(lambda: idx < stop):
            yield idx.value
            idx.value += 1


################################################################################


type _CallableOr[T] = Callable[[], T] | T


@overload
def cond[T: VarT](
    pred: Bool, on_true: _CallableOr[UserValue[T]], on_false: _CallableOr[UserValue[T]]
) -> VarRead[T]: ...
@overload
def cond[V](pred: Bool, on_true: _CallableOr[V], on_false: _CallableOr[V]) -> V: ...


def cond[V](  # pyright: ignore[reportInconsistentOverload]
    pred: Bool, on_true: _CallableOr[V], on_false: _CallableOr[V]
) -> V:
    pred_ = _get(pred)

    true_l = mk_internal_label("cond_true_branch")
    false_l = mk_internal_label("cond_false_branch")
    true_l2 = mk_internal_label("cond_true_branch_2")
    false_l2 = mk_internal_label("cond_false_branch_2")
    end_l = mk_internal_label("cond_end")

    _f.branch(pred_, true_l, false_l)

    def get_val[T](f: _CallableOr[T]) -> T:
        if callable(f):
            return f()  # pyright: ignore[reportReturnType]
        return f

    label(true_l)
    true_out, true_tree = TreeSpec.flatten(get_val(on_true))
    true_vals = [_get(x) for x in true_out]
    jump(true_l2)

    label(false_l)
    false_out, false_tree = TreeSpec.flatten(get_val(on_false))
    false_vals = [_get(x) for x in false_out]
    jump(false_l2)

    out_tree = true_tree.promote_types(false_tree)
    ans_mvars: list[MVar] = [mk_mvar(typ) for typ in out_tree.types]

    label(true_l2)
    for x, mv in zip(true_vals, ans_mvars):
        mv.write(x)
    jump(end_l)

    label(false_l2)
    for x, mv in zip(false_vals, ans_mvars):
        mv.write(x)
    jump(end_l)

    label(end_l)
    return out_tree._unflatten_vals_ro(x.read() for x in ans_mvars)


@overload
def select[T: VarT](pred: Bool, on_true: UserValue[T], on_false: UserValue[T]) -> VarRead[T]: ...
@overload
def select[V](pred: Bool, on_true: V, on_false: V) -> V: ...


def select[V](pred: Bool, on_true: V, on_false: V) -> V:
    pred_ = _get(pred)

    true_vars, true_tree = TreeSpec.flatten(on_true)
    false_vars, false_tree = TreeSpec.flatten(on_false)

    out_tree = true_tree.promote_types(false_tree)
    ans_vars: list[Var] = []

    for x, y, typ in zip(true_vars, false_vars, out_tree.types):
        x_ = _get(x)
        y_ = _get(y)
        ans_vars.append(Select(typ).call(pred_, x_, y_))

    return out_tree._unflatten_vals_ro(ans_vars)


################################################################################


def asm(opcode: str, outputs: Sequence[Variable[Any]], /, *args: UserValue) -> None:
    """
    this feature is in development and may produce incorret output without warning
    """
    for x in outputs:
        assert isinstance(x, Variable)
        assert not x._read_only

    args_ = [_get(x) for x in args]

    instr = RawInstr(
        opcode=opcode,
        in_types=TypeList(get_types(*args_)),
        out_types=TypeList(x._get_type() for x in outputs),
        continues=True,
        _reads=[EffectExternal()],
        _writes=[EffectExternal()],
        jumps=True,
    )
    out_vars = instr.emit(*args_)
    assert len(out_vars) == len(outputs)
    for out_v, uvar in zip(out_vars, outputs):
        uvar._inner.write(out_v)


def asm_fn(opcode: str, /, *args: UserValue) -> VarRead[Any]:
    """
    this feature is in development and may produce incorret output without warning
    """
    out_var = Variable(AnyType)
    asm(opcode, [out_var], *args)
    return out_var.value


AsmBlockLine = LabelLike | tuple[str, *tuple[UserValue, ...]]


def asm_block(*lines_: AsmBlockLine) -> None:
    """
    each argument is a instruction, which is one of:
    (1) (opcode, operands, ...)
        opcode may read all operands and write to any operand Variable that is not read-only
    (2) str | Label (not implemented yet)
    """
    for l in lines_:
        if isinstance(l, LabelLike):
            raise NotImplementedError()

    lines = [x for x in lines_ if isinstance(x, tuple)]

    inputs: list[Value] = []

    def handle_arg(v: UserValue) -> int | MVar:
        if isinstance(v, Variable) and not v._read_only:
            return v._inner
        idx = len(inputs)
        inputs.append(_get(v))
        return idx

    linespecs: list[tuple[str, tuple[MVar | int, ...]]] = []

    for opcode, *args in lines:
        linespecs.append((opcode, tuple(handle_arg(x) for x in args)))

    with track_caller():
        AsmBlock(
            lines=linespecs,
            in_types=TypeList([get_type(x) for x in inputs]),
            out_types=(),
        ).call(*inputs)
