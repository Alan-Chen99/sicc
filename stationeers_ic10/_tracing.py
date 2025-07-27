from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from typing import Concatenate
from typing import Iterator

from ordered_set import OrderedSet
from rich import print

from ._core import Block
from ._core import BoundInstr
from ._core import Fragment
from ._core import InteralBool
from ._core import Label
from ._core import LabelLike
from ._core import MVar
from ._core import Scope
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import get_id
from ._diagnostic import debug_info
from ._functions import branch
from ._functions import jump
from ._functions import unreachable_checked
from ._instructions import EmitLabel
from ._utils import Cell


@dataclass
class Trace:
    fragment: Fragment
    instrs: list[BoundInstr]

    # return_hook: Callable[[Any], None] | None = None


_CUR_SCOPE: Cell[Scope] = Cell()

_CUR_TRACE: Cell[Trace] = Cell()

_EXISTING_EMITTED_LABELS: Cell[set[Label]] = Cell()


def mk_var[T: VarT](typ: type[T]) -> Var[T]:
    """create a Var, tied to the current scope"""
    ans = Var(typ, get_id(), Cell(True), debug_info())
    _CUR_SCOPE.value.vars.add(ans)
    return ans


def ck_val(v: Value) -> None:
    if isinstance(v, Var):
        if not v in _CUR_SCOPE.value.vars:
            raise TypeError("use of out of scope varaible")
        assert v.live.value


def mk_mvar[T: VarT](typ: type[T]) -> MVar[T]:
    ans = MVar(typ, get_id(), debug_info())
    if scope := _CUR_SCOPE.get():
        scope.private_mvars.append(ans)
    return ans


def mk_label(l: LabelLike | None = None, *, implicit: bool = False) -> Label:
    if isinstance(l, Label):
        return l
    if l is None:
        if implicit:
            l = f"_implicit_{get_id()}"
        else:
            l = f"anon_{get_id()}"
    ans = Label(get_id(), l, debug_info(), implicit=implicit)
    if scope := _CUR_SCOPE.get():
        scope.private_labels.add(ans)
    return ans


def mk_internal_label(prefix: str, id: int | None = None) -> Label:
    if id is None:
        id = get_id()
    return mk_label(f"_{prefix}_{id}", implicit=True)


def label(l: LabelLike | None = None, *, implicit: bool = False) -> Label:
    """actually emit the label"""
    l_ = mk_label(l, implicit=implicit)
    assert l_ not in _EXISTING_EMITTED_LABELS.value
    _EXISTING_EMITTED_LABELS.value.add(l_)
    EmitLabel().call(l_)
    return l_


def emit_bound(instr: BoundInstr) -> None:
    for x in instr.inputs:
        ck_val(x)
    _CUR_TRACE.value.instrs.append(instr)


def emit_frag(subf: Fragment) -> None:
    _CUR_TRACE.value.fragment.merge_child(subf)


@contextmanager
def trace_to_fragment(emit: bool = False) -> Iterator[Cell[Fragment]]:
    from ._transforms import normalize

    res: Cell[Fragment] = Cell()

    scope = Scope(
        vars=OrderedSet(()),
        private_mvars=OrderedSet(()),
        private_labels=OrderedSet(()),
    )
    f = Fragment(
        finished_init=False,
        blocks={},
        scope=scope,
    )
    trace = Trace(f, [])
    with _CUR_TRACE.bind(trace), _CUR_SCOPE.bind(scope):
        start = mk_internal_label("frag_fake_start")
        label(start)
        unreachable_checked()
        yield res
        unreachable_checked()

    assert trace.fragment is f
    f.blocks[start] = Block(trace.instrs, debug_info())

    f.finish()
    print("raw traced:")
    print(f)
    f = normalize(f)
    print("after normalize:")
    print(f)
    res.value = f

    if emit:
        emit_frag(f)


def internal_transform[**P, R](
    fn: Callable[Concatenate[Fragment, P], R],
) -> Callable[Concatenate[Fragment, P], R]:
    def inner(frag: Fragment, *args: P.args, **kwargs: P.kwargs) -> R:
        with _CUR_TRACE.bind_clear(), _CUR_SCOPE.bind(frag.scope):
            ans = fn(frag, *args, **kwargs)
        return ans

    return inner


@contextmanager
def trace_main_test() -> Iterator[Cell[Fragment]]:
    id = get_id()
    start = mk_internal_label("main_start", id)
    exit = mk_internal_label("main_exit", id)

    with _EXISTING_EMITTED_LABELS.bind(set()), trace_to_fragment() as res:
        label(start)
        yield res
        jump(exit)


@contextmanager
def trace_if(cond: InteralBool) -> Iterator[None]:
    id = get_id()
    true_branch = mk_internal_label("if_true_branch", id)
    if_end = mk_internal_label("if_end", id)

    branch(cond, true_branch, if_end)

    with trace_to_fragment(emit=True):
        label(true_branch)
        yield
        jump(if_end)

    label(if_end)


################################################################################
