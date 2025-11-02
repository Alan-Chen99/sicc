from __future__ import annotations

from contextlib import AbstractContextManager
from contextlib import contextmanager
from dataclasses import dataclass
from types import NoneType
from typing import Any
from typing import Callable
from typing import Iterator
from typing import overload

from ._api import Bool
from ._api import Int
from ._api import UserValue
from ._api import Variable
from ._api import VarRead
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
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._instructions import Select
from ._instructions import branch
from ._instructions import branch as _branch
from ._tracing import label
from ._tracing import mk_internal_label
from ._tracing import mk_mvar
from ._tracing import trace_to_fragment
from ._tree_utils import TreeSpec
from ._utils import Cell
from ._utils import empty
from ._utils import empty_t
from ._utils import get_id

register_exclusion(__file__)


@dataclass(kw_only=True)
class BlockRef[T]:
    _id: int
    _scope: Scope
    _break_paths: list[tuple[Label, TreeSpec[T], list[MVar], DebugInfo]]
    _out_value: T | empty_t = empty

    exit_label: Label
    finished_tracing: bool

    def break_(self, val: T = None) -> None:
        break_label = mk_internal_label(
            f"block_break_({len(self._break_paths)})",
            self._id,
            scope=self._scope,
        )

        vals, tree = TreeSpec.flatten(val)
        mvars = [mk_mvar(typ) for typ in tree.types]

        for mv, x in zip(mvars, vals):
            mv.write(read_uservalue(x))

        self._break_paths.append((break_label, tree, mvars, debug_info()))
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
def block[T](_type_hint: type[BlockRef[T] | T], /) -> AbstractContextManager[BlockRef[T]]: ...


def block[T](
    _type_hint: type[BlockRef[T] | T] | None = None, /
) -> AbstractContextManager[BlockRef[T]]:
    return block_impl(_type_hint)


@contextmanager
def block_impl[T](_ret_typ: type[BlockRef[T] | T] | None = None) -> Iterator[BlockRef[T]]:
    id = get_id()
    start = mk_internal_label(f"block_start", id)
    end = mk_internal_label(f"block_end", id)

    # outside_scope = top_scope_or_err()

    jump(start)

    try:
        with trace_to_fragment(emit=True) as (scope, _):
            label(start)
            block_ref = BlockRef[T](
                _id=id,
                _scope=scope,
                _break_paths=[],
                exit_label=end,
                finished_tracing=False,
            )
            yield block_ref
            block_ref.finished_tracing = True

            jump(end)

            if len(block_ref._break_paths) == 0:
                return

            out_tree = TreeSpec.promote_types_many(
                *(tree for _, tree, _, _ in block_ref._break_paths)
            )
            out_mvars = [mk_mvar(typ) for typ in out_tree.types]

            for break_label, _, mvars, debug in block_ref._break_paths:
                with add_debug_info(debug):
                    label(break_label)
                    for out_mv, mv in zip(out_mvars, mvars):
                        out_mv.write(mv.read())
                    jump(end)

    finally:
        label(end)

    block_ref._out_value = out_tree.unflatten(Variable(x.type, _mvar=x) for x in out_mvars)


################################################################################

CONTINUE_TO: Cell[Label] = Cell()
BREAK_TO: Cell[Label] = Cell()

ELSE_HOOK: Cell[Callable[[], AbstractContextManager[None]]] = Cell()


@contextmanager
def clear_control_flow_hooks() -> Iterator[None]:
    with (
        CONTINUE_TO.bind_clear(),
        BREAK_TO.bind_clear(),
        ELSE_HOOK.bind_clear(),
    ):
        yield


def continue_():
    cont = CONTINUE_TO.get()
    if cont is None:
        raise RuntimeError("nothing to continue to")
    jump(cont)


def break_():
    br = BREAK_TO.get()
    if br is None:
        raise RuntimeError("nothing to break to")
    jump(br)


def else_() -> AbstractContextManager[None]:
    hook = ELSE_HOOK.get()
    if hook is None:
        raise RuntimeError("not during 'if_'")
    return hook()


################################################################################


@contextmanager
def if_(cond_: Bool, /) -> Iterator[None]:
    cond = read_uservalue(cond_)

    id = get_id()
    true_branch = mk_internal_label("if_true_branch", id)
    false_branch = mk_internal_label("if_false_branch", id)
    if_end = mk_internal_label("if_end", id)

    with track_caller():
        branch(cond, true_branch, false_branch)

    traced_else = Cell(False)

    @contextmanager
    def else_hook() -> Iterator[None]:
        if traced_else.value:
            raise RuntimeError("already traced a else branch")
        traced_else.value = True

        with trace_to_fragment(emit=True):
            label(false_branch)
            yield
            jump(if_end)

    with (
        trace_to_fragment(emit=True),
        ELSE_HOOK.bind(else_hook),
    ):
        label(true_branch)
        yield
        jump(if_end)

    if not traced_else.value:
        with track_caller():
            label(false_branch)
            jump(if_end)

    with track_caller():
        label(if_end)


@contextmanager
def while_(cond_fn: Callable[[], Bool]) -> Iterator[None]:
    id = get_id()
    while_cond = mk_internal_label("while_cond", id)
    while_body = mk_internal_label("while_body", id)
    while_end = mk_internal_label("while_end", id)

    with trace_to_fragment(emit=True):
        with track_caller():
            label(while_cond)
        cond = read_uservalue(cond_fn())
        with track_caller():
            branch(cond, while_body, while_end)

    with (
        trace_to_fragment(emit=True),
        CONTINUE_TO.bind(while_cond),
        BREAK_TO.bind(while_end),
        ELSE_HOOK.bind_clear(),
    ):
        label(while_body)
        yield
        jump(while_cond)

    with track_caller():
        jump(while_cond)
        label(while_end)


def loop() -> AbstractContextManager[None]:
    return while_(lambda: True)


################################################################################


def wrap_iterator[T](it: Iterator[T]) -> Iterator[T]:
    """
    wrap an iterator so that in a `for x in it` loop,
    continue_() and _break() would function correctly
    (instead of continue/break to the internal implementation of the iterator)
    """
    with block(NoneType) as outer:
        for x in it:
            with (
                block(NoneType) as inner,
                BREAK_TO.bind(outer.exit_label),
                CONTINUE_TO.bind(inner.exit_label),
                ELSE_HOOK.bind_clear(),
            ):
                yield x


def wrap_iterator_fn[**P, T](fn: Callable[P, Iterator[T]]) -> Callable[P, Iterator[T]]:
    def inner(*args: P.args, **kwargs: P.kwargs):
        return wrap_iterator(fn(*args, **kwargs))

    return inner


################################################################################


@overload
def range_(stop: Int, /) -> Iterator[VarRead[int]]: ...
@overload
def range_(start: Int, stop: Int, /) -> Iterator[VarRead[int]]: ...


@wrap_iterator_fn
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
    pred_ = read_uservalue(pred)

    true_l = mk_internal_label("cond_true_branch")
    false_l = mk_internal_label("cond_false_branch")
    true_l2 = mk_internal_label("cond_true_branch_2")
    false_l2 = mk_internal_label("cond_false_branch_2")
    end_l = mk_internal_label("cond_end")

    _branch(pred_, true_l, false_l)

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
