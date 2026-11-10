from rich import print as print  # autoflake: skip

from .._core import BoundInstr
from .._core import RegInfo
from .._core import Register
from .._core import Var
from .._core import db_internal
from .._instructions import AddI
from .._instructions import Pop
from .._instructions import Push
from .._instructions import ReadStack
from .._instructions import SplitLifetime
from .._instructions import StackOpChain
from .._instructions import SubI
from .._instructions import WriteStack
from .._tracing import mk_var
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


def _mk_sp_var() -> Var[int]:
    return mk_var(
        int,
        reg=RegInfo(preferred_reg=Register.SP, preferred_weight=1),
    )


def _is_offset(ctx: TransformCtx, addr: Var[int]) -> bool:
    """`addr` is `v + c` or `v - c` for a Var `v` and a constant `c`."""
    d = get_index.call_cached(ctx).vars[addr].def_instr
    if add := d.isinst(AddI):
        return sum(isinstance(x, Var) for x in add.inputs_) == 1
    if sub := d.isinst(SubI):
        v, c = sub.inputs_
        return isinstance(v, Var) and not isinstance(c, Var)
    return False


def _goes_to_sp(ctx: TransformCtx, v: Var) -> bool:
    """`v` is moved into a var that prefers sp, as a coroutine restoring its sp does."""
    index = get_index.call_cached(ctx)
    for use in index.vars[v].uses:
        if split := use.isinst(SplitLifetime):
            (out,) = split.outputs_
            if out.reg.preferred_reg == Register.SP:
                return True
    return False


def _stack_chain_to_push_pop_one(ctx: TransformCtx, instr: BoundInstr[StackOpChain]) -> bool:
    f = ctx.frag

    parts = instr.unpack()

    if len(parts) == 0:
        return False

    if len(parts) == 1:
        # Pointing sp at a single access takes a line, where `get`/`put` take the address
        # directly. That line replaces the one computing an offset address, and disappears
        # when v can live in sp, as a loop iterator over the stack does. A value read into
        # sp overwrites the sp a pop leaves, so the line could never disappear.
        (part,) = parts
        _device, addr, *_val = part.inputs_
        if not isinstance(addr, Var) or not _is_offset(ctx, addr):
            return False
        if (read := part.isinst(ReadStack)) and _goes_to_sp(ctx, read.outputs_[0]):
            return False

    device, addr, *_val = parts[0].inputs_
    for p in parts[1:]:
        assert p.inputs_[0] == device
        assert p.inputs_[1] == addr

    if device != db_internal or not isinstance(addr, Var):
        return False

    if all(p.isinst(WriteStack) for p in parts):
        parts = [p.check_type(WriteStack) for p in parts]

        # we currently dont touch this after tracing
        # will prob change later
        for i, p in enumerate(parts):
            assert p.instr.offset == i

        @f.replace_instr(instr)
        def _():
            sp_cur = _mk_sp_var()

            # yield SplitLifetime(int).bind((sp_cur,), end_addr)

            yield SplitLifetime(int).bind((sp_cur,), addr)

            for p in parts:
                sp_next = _mk_sp_var()
                yield Push.from_parts(
                    WriteStack(p.in_types[2], 0).bind((), db_internal, sp_cur, p.inputs_[2]),
                    AddI().bind((sp_next,), sp_cur, 1),
                )
                sp_cur = sp_next

        return True

    if all(p.isinst(ReadStack) for p in parts):
        parts = [p.check_type(ReadStack) for p in parts]

        # we currently dont touch this after tracing
        # will prob change later
        for i, p in enumerate(parts):
            assert p.instr.offset == i

        @f.replace_instr(instr)
        def _():
            end_addr = mk_var(int)
            yield AddI().bind((end_addr,), addr, len(parts))

            sp_cur = _mk_sp_var()
            yield SplitLifetime(int).bind((sp_cur,), end_addr)

            for p in reversed(parts):
                sp_next = _mk_sp_var()
                yield Pop.from_parts(
                    SubI().bind((sp_next,), sp_cur, 1),
                    ReadStack(p.out_types[0], 0).bind(p.outputs_, db_internal, sp_next),
                )
                sp_cur = sp_next

        return True

    # currently unreachable
    assert False


@LoopingTransform
def stack_chain_to_push_pop(ctx: TransformCtx) -> bool:
    index = get_index.call_cached(ctx)

    for instr in index.instrs:
        if i := instr.isinst(StackOpChain):
            if _stack_chain_to_push_pop_one(ctx, i):
                return True

    return False


@LoopingTransform
def remove_trivial_push_pop(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for instr in index.instrs:
        if i := instr.isinst(Push):
            write, add = i.unpack()
            _device, addr, _val = write.inputs_
            if not isinstance(addr, Var):

                @f.replace_instr(instr)
                def _():
                    yield write
                    yield add

                return True

    return False
