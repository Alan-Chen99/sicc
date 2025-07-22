from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from ordered_set import OrderedSet

from ._core import Block, Bool, BoundInstr, Fragment, Label, LabelLike, get_id, mk_internal_label, mk_label
from ._diagnostic import debug_info
from ._instructions import EmitLabel
from ._utils import Cell
from .functions import branch, jump, unreachable_checked


@dataclass
class Trace:
    fragment: Fragment
    instrs: list[BoundInstr]

    # return_hook: Callable[[Any], None] | None = None


_CUR_TRACE = Cell[Trace]()

_EXISTING_LABELS = Cell[set[Label]]()


def label(l: LabelLike | None = None, *, implicit: bool = False) -> Label:
    """actually emit the label"""
    l_ = mk_label(l, implicit=implicit)
    assert l_ not in _EXISTING_LABELS.value
    _EXISTING_LABELS.value.add(l_)
    EmitLabel().call(l_)
    return l_


def mark_label_private(l: Label) -> None:
    if tr := _CUR_TRACE.get():
        tr.fragment.private_labels.append(l)


def emit_bound(instr: BoundInstr[Any, Any]) -> None:
    _CUR_TRACE.value.instrs.append(instr)


def emit_frag(subf: Fragment) -> None:
    _CUR_TRACE.value.fragment.merge_child(subf)


@contextmanager
def trace_to_fragment(emit: bool = False) -> Iterator[Cell[Fragment]]:
    res: Cell[Fragment] = Cell()

    f = Fragment(
        finished_init=False,
        blocks={},
        private_mvars=OrderedSet(()),
        private_labels=OrderedSet(()),
    )
    trace = Trace(f, [])
    with _CUR_TRACE.bind(trace):
        start = mk_internal_label("frag_fake_start")
        label(start)
        unreachable_checked()
        yield res
        unreachable_checked()

    assert trace.fragment is f
    f.blocks[start] = Block(trace.instrs, debug_info())

    f.finish()
    res.value = f

    if emit:
        emit_frag(f)


@contextmanager
def trace_main_test() -> Iterator[Cell[Fragment]]:
    id = get_id()
    start = mk_internal_label("main_start", id)
    exit = mk_internal_label("main_exit", id)

    with _EXISTING_LABELS.bind(set()), trace_to_fragment() as res:
        label(start)
        yield res
        jump(exit)


@contextmanager
def if_(cond: Bool) -> Iterator[None]:
    id = get_id()
    true_branch = mk_internal_label("true_branch", id)
    if_end = mk_internal_label("if_end", id)

    branch(cond, true_branch, if_end)

    with trace_to_fragment(emit=True):
        label(true_branch)
        yield
        jump(if_end)

    label(if_end)


# class Subr:
#     start: Label
#     exit: Label
#     frag: Fragment
#     # return_val
