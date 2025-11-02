from __future__ import annotations

from contextlib import AbstractContextManager
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Iterator
from typing import overload

from . import _functions as _f
from ._api import Bool
from ._api import Int
from ._api import TreeSpec
from ._api import UserValue
from ._api import ValLabelLike
from ._api import Variable
from ._api import VarRead
from ._api import _get_label
from ._api import copy_tree
from ._api import jump
from ._api import read_uservalue
from ._core import Label
from ._core import MVar
from ._core import Scope
from ._core import Var
from ._core import VarT
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import debug_info
from ._instructions import Select
from ._state import State
from ._tracing import _CUR_SCOPE
from ._tracing import break_
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import mk_mvar
from ._tracing import trace_if
from ._tracing import trace_while
from ._utils import empty
from ._utils import empty_t
from ._utils import get_id


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


################################################################################


def branch(cond: Bool, on_true: ValLabelLike, on_false: ValLabelLike) -> None:
    return _f.branch(read_uservalue(cond), _get_label(on_true), _get_label(on_false))


def cjump(cond: Bool, on_true: ValLabelLike) -> None:
    cont = mk_internal_label("cjump_cont")
    _f.branch(read_uservalue(cond), _get_label(on_true), cont)
    label(cont)


def if_(cond: Bool) -> AbstractContextManager[None]:
    return trace_if(read_uservalue(cond))


def while_(cond_fn: Callable[[], Bool]) -> AbstractContextManager[None]:
    def inner():
        return read_uservalue(cond_fn())

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
    pred_ = read_uservalue(pred)

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
    true_vals = [read_uservalue(x) for x in true_out]
    jump(true_l2)

    label(false_l)
    false_out, false_tree = TreeSpec.flatten(get_val(on_false))
    false_vals = [read_uservalue(x) for x in false_out]
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
    pred_ = read_uservalue(pred)

    true_vars, true_tree = TreeSpec.flatten(on_true)
    false_vars, false_tree = TreeSpec.flatten(on_false)

    out_tree = true_tree.promote_types(false_tree)
    ans_vars: list[Var] = []

    for x, y, typ in zip(true_vars, false_vars, out_tree.types):
        x_ = read_uservalue(x)
        y_ = read_uservalue(y)
        ans_vars.append(Select(typ).call(pred_, x_, y_))

    return out_tree._unflatten_vals_ro(ans_vars)
