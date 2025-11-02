from __future__ import annotations

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
from ._core import get_type
from ._diagnostic import DebugInfo
from ._diagnostic import add_debug_info
from ._diagnostic import clear_debug_info
from ._diagnostic import debug_info
from ._diagnostic import describe_fn
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._instructions import Bundle
from ._instructions import EmitLabel
from ._instructions import EndPlaceholder
from ._instructions import jump
from ._instructions import unreachable_checked
from ._utils import ByIdMixin
from ._utils import Cell
from ._utils import get_id
from .config import verbose
from .config import with_status

register_exclusion(__file__)


@dataclass
class Trace:
    fragment: Fragment
    instrs: list[BoundInstr]


SCOPE_STACK: Cell[list[Scope]] = Cell([])

_CUR_TRACE: Cell[Trace] = Cell()

_EXISTING_EMITTED_LABELS: Cell[set[Label]] = Cell(set())


def top_scope_or_err() -> Scope:
    if len(SCOPE_STACK.value) == 0:
        raise RuntimeError("cannot do this without being inside a function")
    return SCOPE_STACK.value[-1]


def mk_var[T: VarT](
    typ: type[T], *, reg: RegInfo = RegInfo(), debug: DebugInfo | None = None
) -> Var[T]:
    """create a Var, tied to the current scope"""
    ans = Var(typ, get_id(), Cell(True), reg, debug or debug_info())
    top_scope_or_err().vars.add(ans)
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
    scoped: bool = False,
    reg: RegInfo = RegInfo(),
    debug: DebugInfo | None = None,
) -> MVar[T]:
    ans = MVar(typ, get_id(), reg, debug or debug_info(1))
    if scoped:
        top_scope_or_err().scoped_mvars.append(ans)
    return ans


def ensure_label(l: LabelLike | None = None, unique: bool = False) -> Label:
    if isinstance(l, Label):
        return l
    if l is None:
        l = f"anon_{get_id()}"
    elif unique:
        l = f"{l}_{get_id()}"
    ans = Label(l, debug_info(), implicit=False)
    return ans


def mk_internal_label(
    prefix: str, id: int | None = None, private: bool = True, scope: Scope | None = None
) -> Label:
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
    if scope:
        scope.private_labels.add(ans)
    elif private:
        top_scope_or_err().private_labels.add(ans)
    return ans


def label(l: LabelLike | None = None, *, unique: bool = False, implicit: bool = False) -> Label:
    """actually emit the label"""
    l_ = ensure_label(l, unique=unique)
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


_out_of_scope_mvars: Cell[set[MVar]] = Cell(set())


def check_mvar_scope(v: MVar):
    pass


@contextmanager
def trace_to_fragment(
    emit: bool = False,
    optimize: bool = True,
) -> Iterator[tuple[Scope, Cell[Fragment]]]:
    from ._transforms import optimize_frag
    from ._transforms.control_flow import compute_label_provenance

    res: Cell[Fragment] = Cell()

    scope = Scope(
        vars=OrderedSet(()),
        scoped_mvars=OrderedSet(()),
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
        # continue_to=continue_to,
        # break_to=break_to,
        # else_hook=else_hook,
    )
    with (
        clear_debug_info(),
        _CUR_TRACE.bind(trace),
        SCOPE_STACK.bind(SCOPE_STACK.value + [scope]),
    ):
        start = mk_internal_label("frag_fake_start")
        label(start)
        unreachable_checked()
        yield scope, res
        unreachable_checked()

    assert trace.fragment is f
    f.blocks[start] = Block(trace.instrs, debug_info())

    f.finish()
    _out_of_scope_mvars.value |= scope.scoped_mvars

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
    with _CUR_TRACE.bind_clear(), SCOPE_STACK.bind([frag.scope]), clear_debug_info():
        yield


PROGRAM_EXIT_TO: Cell[Label] = Cell()


@contextmanager
def trace_program() -> Iterator[Cell[TracedProgram]]:
    from ._transforms import global_opts
    from ._transforms.basic import mark_all_private_except

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
        trace_to_fragment() as (_, res),
    ):
        label(start)
        yield ans

        label(exit)
        EndPlaceholder().call()

        for x in subrs:
            emit_frag(x.frag)

    f = res.value
    mark_all_private_except(f, [start])
    global_opts(f)

    if verbose.value >= 1:
        print("after optimize with marked private:")
        print(f)

    ans.value = TracedProgram(start, f)


def exit_program() -> None:
    jump(PROGRAM_EXIT_TO.value)


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

        with track_caller():
            for arg, argvar in zip(args, self.arg_mvars):
                argvar.write(arg)

        self.ra_mvar.write(l)
        with track_caller():
            jump(self.start)

        label(l)
        return tuple(x.read() for x in self.ret_mvars)


SUBR_CALL_HOOK: Cell[Callable[[RawSubr], None]] = Cell()


def trace_to_raw_subr(arg_types: VarTS, fn: Callable[[*tuple[Var, ...]], tuple[Value, ...]]):
    ra_mvar = mk_mvar(Label)
    arg_mvars = tuple(mk_mvar(x) for x in arg_types)

    start = mk_internal_label(f"[{describe_fn(fn)}]", private=False)

    with with_status(describe_fn(fn)), trace_to_fragment() as (_scope, res):
        with add_debug_info(DebugInfo(describe=f"<function {describe_fn(fn)}>")):
            label(start)

        # allow spilling ra into another reg at function start
        ra_mvar_inner = mk_mvar(Label)
        ra_mvar_inner.write(ra_mvar.read())

        with add_debug_info(DebugInfo(describe=f"args of {describe_fn(fn)}")):
            argvals = tuple(x.read() for x in arg_mvars)
        out_vars = fn(*argvals)
        ret_mvars = tuple(mk_mvar(get_type(x)) for x in out_vars)
        for mv, v in zip(ret_mvars, out_vars):
            with add_debug_info(DebugInfo(describe=f"return value from {describe_fn(fn)}")):
                mv.write(v)
        with add_debug_info(DebugInfo(describe=f"return from {describe_fn(fn)}")):
            jump(ra_mvar_inner.read())

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
    from ._transforms import global_opts
    from ._transforms.fuse_blocks import force_fuse_into_one
    from ._transforms.fuse_blocks import fuse_blocks_all

    start = mk_internal_label(f"isolate")

    with trace_to_fragment() as (_, res):
        label(start)
        yield
        EndPlaceholder().call()

    frag = res.value
    global_opts(frag)
    fuse_blocks_all(frag)

    force_fuse_into_one(frag, start)

    block = frag.blocks[start]
    Bundle.from_block(block).emit()
