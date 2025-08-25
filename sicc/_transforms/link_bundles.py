from rich import print as print  # autoflake: skip

from .._core import AlwaysUnpack
from .._core import Block
from .._core import BoundInstr
from .._core import Fragment
from .._core import InternalValLabel
from .._core import Label
from .._core import MVar
from .._core import RegInfo
from .._core import Register
from .._core import Var
from .._core import VirtualConst
from .._core import WriteMVar
from .._diagnostic import add_debug_info
from .._instructions import CondJump
from .._instructions import CondJumpAndLink
from .._instructions import EmitLabel
from .._instructions import EndPlaceholder
from .._instructions import Jump
from .._instructions import JumpAndLink
from .._instructions import PredBranch
from .._instructions import PredNAN
from .._instructions import PredNotNAN
from .._tracing import mk_internal_label
from .._tracing import mk_mvar
from .basic import get_index
from .basic import map_mvars
from .basic import split_blocks
from .fuse_blocks import fuse_blocks_trivial_jumps
from .optimize_mvars import compute_mvar_lifetime
from .optimize_mvars import support_mvar_analysis
from .utils import LoopingTransform
from .utils import TransformCtx


def _is_suitable_as_ra(ctx: TransformCtx, mv: MVar):
    index = get_index.call_cached(ctx)

    if not index.mvars[mv].private:
        return False
    if mv.reg.force_reg:
        return False
    if mv.reg.preferred_reg is not None and mv.reg.preferred_reg != Register.RA:
        return False
    if not support_mvar_analysis(ctx, mv, AlwaysUnpack()):
        return False

    return True


def _mark_prefer_ra(f: Fragment, mv: MVar):
    rep_mv = mk_mvar(
        mv.type,
        reg=RegInfo(preferred_reg=Register.RA, preferred_weight=mv.reg.preferred_weight + 1),
        debug=mv.debug,
    )
    map_mvars(f, {mv: rep_mv})


def _try_pack_call_one(
    ctx: TransformCtx,
    block: Block,
    write_instr: BoundInstr[WriteMVar[Label]],
    jump_instr: BoundInstr[Jump],
) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    assert block.end == jump_instr

    (ret_label,) = write_instr.inputs_

    if isinstance(ret_label, Var | VirtualConst):
        return False

    l = index.labels[ret_label]

    # check if possible to fuse ret_label block after jump
    # if not making link instr does not acomplish anything
    if not l.private:
        return False
    if list(l.uses) != [write_instr]:
        return False
    assert ret_label in f.blocks  # bc of split_blocks
    if (
        f.blocks[ret_label].end.isinst(EndPlaceholder)
        and not index.labels[block.label].private
        and len(f.blocks) >= 3
    ):
        # we wont be able to fuse the start block and exit block
        return False

    mv = write_instr.instr.s
    if not _is_suitable_as_ra(ctx, mv):
        return False
    res = compute_mvar_lifetime(ctx, mv, AlwaysUnpack())

    if (info := res.reachable.get(jump_instr)) and info.possible_defs == [write_instr]:
        # valid to write label to mv before instr
        pass
    else:
        return False

    @f.replace_instr(jump_instr)
    def _():
        tmp_label = mk_internal_label(ret_label.id)

        with add_debug_info(write_instr.debug):
            new_write = WriteMVar(mv).bind((), tmp_label)

        yield JumpAndLink.from_parts(
            new_write,
            jump_instr,
            EmitLabel().bind((), tmp_label),
        )
        yield Jump().bind((), ret_label)

    f.replace_instr(write_instr)(lambda: [])

    _mark_prefer_ra(f, mv)

    return True


@LoopingTransform
def pack_call(ctx: TransformCtx) -> bool:
    f = ctx.frag

    if split_blocks(f):
        return True

    for b in f.blocks.values():
        if end_instr := b.end.isinst(Jump):
            for instr in reversed(b.body):
                if (instr := instr.isinst(WriteMVar)) and instr.instr.s.type == Label:
                    if _try_pack_call_one(ctx, b, instr, end_instr):
                        # print("end of pack:")
                        # print(f)
                        fuse_blocks_trivial_jumps(f)
                        return True

    return False


def _try_predbranch_one(
    ctx: TransformCtx, instr: BoundInstr[PredBranch], ret_label: InternalValLabel
) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    pred, branch = instr.unpack()

    if pred.isinst((PredNAN, PredNotNAN)):
        # these dont have b-al instructions
        return False

    if not isinstance(ret_label, Label):
        return False

    l = index.labels[ret_label]

    if not l.private:
        return False

    other_uses = list(l.uses - {instr})
    if len(other_uses) == 1 and (write_instr := other_uses[0].isinst(WriteMVar)):
        pass
    else:
        return False

    assert write_instr.inputs_ == (ret_label,)

    # try to move write_instr to before instr

    mv = write_instr.instr.s
    if not _is_suitable_as_ra(ctx, mv):
        return False
    res = compute_mvar_lifetime(ctx, mv, AlwaysUnpack())

    if (info := res.reachable.get(branch)) and info.possible_defs == [write_instr]:
        # valid to write label to mv before instr
        pass
    else:
        return False

    # make the change
    predvar, l_t, l_f = branch.inputs_
    # (pred_var,) = instr.outputs_

    @f.replace_instr(instr)
    def _():
        tmp_label = mk_internal_label(ret_label.id)

        if l_t == ret_label:
            yield CondJumpAndLink.from_parts(
                WriteMVar(mv).bind((), tmp_label),
                pred.instr.negate(pred),
                CondJump().bind((), predvar, l_f),
                EmitLabel().bind((), tmp_label),
            )
        if l_f == ret_label:
            yield CondJumpAndLink.from_parts(
                WriteMVar(mv).bind((), tmp_label),
                pred,
                CondJump().bind((), predvar, l_t),
                EmitLabel().bind((), tmp_label),
            )

        yield Jump().bind((), ret_label)

    _mark_prefer_ra(f, mv)

    return True


@LoopingTransform
def pack_cond_call(ctx: TransformCtx) -> bool:
    f = ctx.frag

    for b in f.blocks.values():
        if instr := b.end.isinst(PredBranch):
            _, l_t, l_f = instr.unpack()[1].inputs_
            if _try_predbranch_one(ctx, instr, l_t):
                return True
            if _try_predbranch_one(ctx, instr, l_f):
                return True

    return False
