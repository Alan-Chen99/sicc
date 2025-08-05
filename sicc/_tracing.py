from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from typing import Iterator

from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip

from ._core import Block
from ._core import BoundInstr
from ._core import Fragment
from ._core import InteralBool
from ._core import Label
from ._core import LabelLike
from ._core import LowerRes
from ._core import MVar
from ._core import Scope
from ._core import Value
from ._core import Var
from ._core import VarT
from ._core import VarTS
from ._core import can_cast_implicit_many
from ._core import get_id
from ._core import get_types
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

# register_exclusion(__file__)


@dataclass
class Trace:
    fragment: Fragment
    instrs: list[BoundInstr]

    # return_hook: Callable[[Any], None] | None = None


_CUR_SCOPE: Cell[Scope] = Cell()

_CUR_TRACE: Cell[Trace] = Cell()

_EXISTING_EMITTED_LABELS: Cell[set[Label]] = Cell(set())

_EMIT_HOOK: Cell[Callable[[BoundInstr], None]] = Cell()


def mk_var[T: VarT](typ: type[T], *, debug: DebugInfo | None = None) -> Var[T]:
    """create a Var, tied to the current scope"""
    ans = Var(typ, get_id(), Cell(True), debug or debug_info())
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
    typ: type[T], *, force_public: bool = False, debug: DebugInfo | None = None
) -> MVar[T]:
    ans = MVar(typ, get_id(), debug or debug_info(1))
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


def _emit_bound_default(instr: BoundInstr) -> None:
    for x in instr.inputs:
        ck_val(x)
    _CUR_TRACE.value.instrs.append(instr)


def emit_bound(instr: BoundInstr) -> None:
    _EMIT_HOOK.value(instr)


def emit_frag(subf: Fragment) -> None:
    _CUR_TRACE.value.fragment.merge_child(subf)


@contextmanager
def trace_to_fragment(emit: bool = False, optimize: bool = True) -> Iterator[Cell[Fragment]]:
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
    trace = Trace(f, [])
    with (
        clear_debug_info(),
        _CUR_TRACE.bind(trace),
        _CUR_SCOPE.bind(scope),
        _EMIT_HOOK.bind(_emit_bound_default),
    ):
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
    if optimize:
        optimize_frag(f)
        print("after optimize:")
        print(f)
    res.value = f

    if emit:
        emit_frag(f)


@contextmanager
def internal_trace_to_instrs() -> Iterator[Cell[list[BoundInstr]]]:
    assert _CUR_SCOPE.get() is not None

    res: Cell[list[BoundInstr]] = Cell()
    ans: list[BoundInstr] = []

    def emit_hook(x: BoundInstr):
        ans.append(x)

    with _EMIT_HOOK.bind(emit_hook), _CUR_TRACE.bind_clear():
        yield res

    res.value = ans


def internal_trace_as_rep(
    prev_instr: BoundInstr, fn: Callable[[*tuple[Value, ...]], LowerRes]
) -> list[BoundInstr]:
    with internal_trace_to_instrs() as res:
        out_vars = fn(*prev_instr.inputs)
        if out_vars is None:
            out_vars = ()
        elif isinstance(out_vars, Var):
            out_vars = (out_vars,)
        else:
            out_vars = tuple(out_vars)

    instrs = res.value

    assert can_cast_implicit_many(get_types(*out_vars), get_types(*prev_instr.outputs))

    # substitude back previous output vars
    # FIXME: what if "fn" just returned one of its inputs?
    def gen():
        for instr in instrs:
            for x, y in zip(out_vars, prev_instr.outputs):
                instr = instr.sub_val(x, y, inputs=False, outputs=True, strict=True)
            yield instr

    return list(gen())


@contextmanager
def internal_transform(frag: Fragment) -> Iterator[None]:
    if _CUR_TRACE.get() is None and _CUR_SCOPE.get() is frag.scope:
        yield
        return

    with _CUR_TRACE.bind_clear(), _CUR_SCOPE.bind(frag.scope):
        yield


@contextmanager
def trace_main_test() -> Iterator[Cell[Fragment]]:
    from ._transforms import global_opts
    from ._transforms import optimize_frag
    from ._transforms.basic import mark_all_private_except

    id = get_id()
    start = mk_internal_label("program_start", id, private=False)
    exit = mk_internal_label("program_exit", id, private=False)

    subrs: OrderedSet[RawSubr] = OrderedSet(())

    def subr_hook(subr: RawSubr):
        subrs.add(subr)

    # with _EXISTING_EMITTED_LABELS.bind(set()), trace_to_fragment() as res:
    with SUBR_CALL_HOOK.bind(subr_hook), trace_to_fragment() as res:
        label(start)
        yield res
        jump(exit)

        for x in subrs:
            emit_frag(x.frag)

    f = res.value
    mark_all_private_except(f, [start, exit])
    optimize_frag(f)

    print("after optimize with marked private:")
    print(f)

    global_opts(f)

    print("after global optimize:")
    print(f)


@contextmanager
def trace_if(cond: InteralBool) -> Iterator[None]:
    id = get_id()
    true_branch = mk_internal_label("if_true_branch", id)
    if_end = mk_internal_label("if_end", id)

    with track_caller():
        branch(cond, true_branch, if_end)

    with trace_to_fragment(emit=True):
        label(true_branch)
        yield
        jump(if_end)

    with track_caller():
        label(if_end)


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
        jump(self.start)

        label(l)
        return tuple(x.read() for x in self.ret_mvars)


SUBR_CALL_HOOK: Cell[Callable[[RawSubr], None]] = Cell()


def trace_to_raw_subr(arg_types: VarTS, fn: Callable[[*tuple[Var, ...]], tuple[Var, ...]]):
    # should not matter but safer
    with _CUR_TRACE.bind_clear(), _CUR_SCOPE.bind_clear(), _EMIT_HOOK.bind_clear():
        ra_mvar = mk_mvar(Label, force_public=True)
        arg_mvars = tuple(mk_mvar(x, force_public=True) for x in arg_types)

        start = mk_internal_label(f"[{describe_fn(fn)}]", private=False)

        with trace_to_fragment() as res:
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
    from ._transforms.fuse_blocks import force_fuse_into_one

    start = mk_internal_label(f"isolate")

    with trace_to_fragment() as res:
        label(start)
        yield
        EndPlaceholder().call()

    frag = res.value
    force_fuse_into_one(frag, start)

    block = frag.blocks[start]
    Bundle.from_block(block).emit()
