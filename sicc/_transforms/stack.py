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


def _stack_chain_to_push_pop_one(ctx: TransformCtx, instr: BoundInstr[StackOpChain]) -> bool:
    f = ctx.frag

    parts = instr.unpack()

    if len(parts) == 0:
        return False

    if len(parts) == 1:
        (part,) = parts
        _pin, addr, *_val = part.inputs_
        if not isinstance(addr, Var):
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
