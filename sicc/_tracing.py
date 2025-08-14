from __future__ import annotations

from contextlib import AbstractContextManager
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from typing import Iterator

from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip

from ._core import FORMAT_ANNOTATE
from ._core import Block
from ._core import BoundInstr
from ._core import Fragment
from ._core import InteralBool
from ._core import Label
from ._core import LabelLike
from ._core import MVar
from ._core import RegInfo
from ._core import Scope
from ._core import TracedProgram
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VarTS
from ._core import get_id
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import clear_debug_info
from ._diagnostic import debug_info
from ._diagnostic import describe_fn
from ._diagnostic import track_caller
from ._functions import branch
from ._functions import jump
from ._functions import unreachable_checked
from ._instructions import Bundle
from ._instructions import EmitLabel
from ._instructions import EndPlaceholder
from ._utils import ByIdMixin
from ._utils import Cell
from .config import verbose
from .config import with_status

# register_exclusion(__file__)


@dataclass
class Trace:
    fragment: Fragment
    instrs: list[BoundInstr]

    continue_to: Label | None = None
    break_to: Label | None = None

    else_hook: Callable[[], AbstractContextManager[None]] | None = None


_CUR_SCOPE: Cell[Scope] = Cell()

_CUR_TRACE: Cell[Trace] = Cell()

_EXISTING_EMITTED_LABELS: Cell[set[Label]] = Cell(set())


def mk_var[T: VarT](
    typ: type[T], *, reg: RegInfo = RegInfo(), debug: DebugInfo | None = None
) -> Var[T]:
    """create a Var, tied to the current scope"""
    ans = Var(typ, get_id(), Cell(True), reg, debug or debug_info())
    _CUR_SCOPE.value.vars.add(ans)
    return ans


def ck_val(v: Value) -> None:
    # FIXME: this check has been temporarily disabled since introduction of bundles
    # this is bc previously BoundInstr is only created under specific places
    # and now they can be created almost anywhere from a .bundles function
    # it should be moved somewhere
    pass
    # if isinstance(v, Var):
    #     if not v in _CUR_SCOPE.value.vars:
    #         raise TypeError("use of out of scope varaible")
    #     assert v.live.value


def mk_mvar[T: VarT](
    typ: type[T],
    *,
    force_public: bool = False,
    reg: RegInfo = RegInfo(),
    debug: DebugInfo | None = None,
) -> MVar[T]:
    ans = MVar(typ, get_id(), reg, debug or debug_info(1))
    if not force_public and (scope := _CUR_SCOPE.get()):
        scope.private_mvars.append(ans)
    return ans


def ensure_label(l: LabelLike | None = None) -> Label:
    if isinstance(l, Label):
        return l
    if l is None:
        l = f"anon_{get_id()}"
    ans = Label(l, debug_info(), implicit=False)
    return ans


def mk_internal_label(prefix: str, id: int | None = None, private: bool = True) -> Label:
    if id is None:
        id = get_id()

    def get_prefix():
        if not prefix.startswith("_"):
            return prefix
        last = prefix.split("_")[-1]
        try:
            int(last)
        except ValueError:
            return prefix
        return prefix.removeprefix("_").removesuffix("_" + last)

    ans = Label(f"_{get_prefix()}_{id}", debug_info(), implicit=True)
    if private:
        _CUR_SCOPE.value.private_labels.add(ans)
    return ans


def label(l: LabelLike | None = None, *, implicit: bool = False) -> Label:
    """actually emit the label"""
    l_ = ensure_label(l)
    assert l_ not in _EXISTING_EMITTED_LABELS.value
    _EXISTING_EMITTED_LABELS.value.add(l_)
    # with add_debug_info(DebugInfo(show_src=True)):
    with track_caller():
        EmitLabel().call(l_)
    return l_


def emit_bound(instr: BoundInstr) -> None:
    for x in instr.inputs:
        ck_val(x)
    _CUR_TRACE.value.instrs.append(instr)


def emit_frag(subf: Fragment) -> None:
    _CUR_TRACE.value.fragment.merge_child(subf)


@contextmanager
def trace_to_fragment(
    emit: bool = False,
    optimize: bool = True,
    continue_to: Label | None = None,
    break_to: Label | None = None,
    else_hook: Callable[[], AbstractContextManager[None]] | None = None,
) -> Iterator[Cell[Fragment]]:
    from ._transforms import compute_label_provenance
    from ._transforms import optimize_frag

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
    trace = Trace(
        f,
        [],
        continue_to=continue_to,
        break_to=break_to,
        else_hook=else_hook,
    )
    with (
        clear_debug_info(),
        _CUR_TRACE.bind(trace),
        _CUR_SCOPE.bind(scope),
    ):
        start = mk_internal_label("frag_fake_start")
        label(start)
        unreachable_checked()
        yield res
        unreachable_checked()

    assert trace.fragment is f
    f.blocks[start] = Block(trace.instrs, debug_info())

    f.finish()
    if verbose.value >= 1:
        print("raw traced:")
        if verbose.value >= 2:
            with FORMAT_ANNOTATE.bind(compute_label_provenance(f).annotate):
                print(f)
        else:
            print(f)
    if optimize:
        optimize_frag(f)
        if verbose.value >= 1:
            print("after optimize:")
            print(f)
    res.value = f

    if emit:
        emit_frag(f)


@contextmanager
def internal_transform(frag: Fragment) -> Iterator[None]:
    if _CUR_TRACE.get() is None and _CUR_SCOPE.get() is frag.scope:
        with clear_debug_info():
            yield
        return

    with _CUR_TRACE.bind_clear(), _CUR_SCOPE.bind(frag.scope), clear_debug_info():
        yield


PROGRAM_EXIT_TO: Cell[Label] = Cell()


@contextmanager
def trace_program() -> Iterator[Cell[TracedProgram]]:
    from ._transforms import optimize_frag
    from ._transforms.basic import mark_all_private_except
    from ._transforms.utils import frag_is_global

    ans: Cell[TracedProgram] = Cell()

    id = get_id()
    start = mk_internal_label("program_start", id, private=False)
    exit = mk_internal_label("program_exit", id, private=False)

    subrs: OrderedSet[RawSubr] = OrderedSet(())

    def subr_hook(subr: RawSubr):
        subrs.add(subr)

    with (
        SUBR_CALL_HOOK.bind(subr_hook),
        PROGRAM_EXIT_TO.bind(exit),
        trace_to_fragment() as res,
    ):
        label(start)
        yield ans

        label(exit)
        EndPlaceholder().call()

        for x in subrs:
            emit_frag(x.frag)

    f = res.value
    mark_all_private_except(f, [start])
    with frag_is_global.bind(True):
        optimize_frag(f)

    if verbose.value >= 1:
        print("after optimize with marked private:")
        print(f)

    ans.value = TracedProgram(start, f)


def continue_():
    cont = _CUR_TRACE.value.continue_to
    if cont is None:
        raise RuntimeError("nothing to continue to")
    jump(cont)


def break_():
    br = _CUR_TRACE.value.break_to
    if br is None:
        raise RuntimeError("nothing to break to")
    jump(br)


def else_() -> AbstractContextManager[None]:
    hook = _CUR_TRACE.value.else_hook
    if hook is None:
        raise RuntimeError("not during 'if_'")
    return hook()


def exit_program() -> None:
    jump(PROGRAM_EXIT_TO.value)


@contextmanager
def trace_if(cond: InteralBool) -> Iterator[None]:
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

        with trace_to_fragment(
            emit=True,
            continue_to=_CUR_TRACE.value.continue_to,
            break_to=_CUR_TRACE.value.break_to,
        ):
            label(false_branch)
            yield
            jump(if_end)

    with trace_to_fragment(
        emit=True,
        continue_to=_CUR_TRACE.value.continue_to,
        break_to=_CUR_TRACE.value.break_to,
        else_hook=else_hook,
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
def trace_while(cond_block: Callable[[], InteralBool]) -> Iterator[None]:
    id = get_id()
    while_cond = mk_internal_label("while_cond", id)
    while_body = mk_internal_label("while_body", id)
    while_end = mk_internal_label("while_end", id)

    with trace_to_fragment(emit=True):
        with track_caller():
            label(while_cond)
        cond = cond_block()
        with track_caller():
            branch(cond, while_body, while_end)

    with trace_to_fragment(
        emit=True,
        continue_to=while_cond,
        break_to=while_end,
    ):
        label(while_body)
        yield
        jump(while_cond)

    with track_caller():
        jump(while_cond)
        label(while_end)


################################################################################


@dataclass(eq=False)
class RawSubr(ByIdMixin):
    """
    non-recursive subroutine
    """

    id: int

    frag: Fragment
    start: Label

    ra_mvar: MVar[Label]
    arg_mvars: tuple[MVar, ...]
    ret_mvars: tuple[MVar, ...]

    def call(self, *args: Value) -> tuple[Var, ...]:
        SUBR_CALL_HOOK.value(self)

        assert len(args) == len(self.arg_mvars)

        l = mk_internal_label("call_ret")

        for arg, argvar in zip(args, self.arg_mvars):
            argvar.write(arg)

        self.ra_mvar.write(l)
        with track_caller():
            jump(self.start)

        label(l)
        return tuple(x.read() for x in self.ret_mvars)


SUBR_CALL_HOOK: Cell[Callable[[RawSubr], None]] = Cell()


def trace_to_raw_subr(arg_types: VarTS, fn: Callable[[*tuple[Var, ...]], tuple[Var, ...]]):
    # should not matter but safer
    with _CUR_TRACE.bind_clear(), _CUR_SCOPE.bind_clear():
        ra_mvar = mk_mvar(Label, force_public=True)
        arg_mvars = tuple(mk_mvar(x, force_public=True) for x in arg_types)

        start = mk_internal_label(f"[{describe_fn(fn)}]", private=False)

        with with_status(describe_fn(fn)), trace_to_fragment() as res:
            with add_debug_info(DebugInfo(describe=repr(fn))):
                label(start)
            with add_debug_info(DebugInfo(describe=f"args of {describe_fn(fn)}")):
                argvals = tuple(x.read() for x in arg_mvars)
            out_vars = fn(*argvals)
            ret_mvars = tuple(mk_mvar(x.type, force_public=True) for x in out_vars)
            for mv, v in zip(ret_mvars, out_vars):
                with add_debug_info(DebugInfo(describe=f"return value from {describe_fn(fn)}")):
                    mv.write(v)
            with add_debug_info(
                DebugInfo(describe=f"jump to return address from {describe_fn(fn)}")
            ):
                jump(ra_mvar.read())

        return RawSubr(
            get_id(),
            frag=res.value,
            start=start,
            ra_mvar=ra_mvar,
            arg_mvars=arg_mvars,
            ret_mvars=ret_mvars,
        )


@contextmanager
def trace_bundle() -> Iterator[None]:
    """for debugging/testing purposes"""
    from ._transforms import fuse_blocks_all
    from ._transforms import global_opts
    from ._transforms.fuse_blocks import force_fuse_into_one

    start = mk_internal_label(f"isolate")

    with trace_to_fragment() as res:
        label(start)
        yield
        EndPlaceholder().call()

    frag = res.value
    global_opts(frag)
    fuse_blocks_all(frag)

    force_fuse_into_one(frag, start)

    block = frag.blocks[start]
    Bundle.from_block(block).emit()
