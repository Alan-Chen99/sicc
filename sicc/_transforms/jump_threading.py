"""
Jump-threading transforms.

This module contains a small, targeted jump-threading pass for two patterns:

- the short-circuit lowering produced for boolean `and` / `or`: a predecessor block `pred`
  branches on `cond` into an intermediate block `mid` that immediately branches on a value
  already known at the end of `pred`;
- `pred` sets an MVar to a label and jumps to a block `mid` that only jumps through that MVar:
  the first step of a static task, whose state holds its start label, or a call to an empty
  subroutine, whose return address holds the label after the call.

In both, the pass rewrites the `pred -> mid` edge to jump directly to the final target.

Safety constraints:

- `mid` must be "trivial": its body is side-effect-free apart from writing back a value just
  read from the same MVar, and any vars defined in `mid` are only used within `mid` (so
  bypassing `mid` does not remove needed definitions).
- For the MVar reload case, the lifetime analysis must show a single reaching `WriteMVar`
  at `pred` with no external path.

Limitations: the pass only runs on global fragments (see `frag_is_global`).
"""

from __future__ import annotations

from .._core import Block
from .._core import BoundInstr
from .._core import Label
from .._core import ReadMVar
from .._core import Value
from .._core import Var
from .._core import WriteMVar
from .._instructions import Branch
from .._instructions import Jump
from .basic import Index
from .basic import get_index
from .optimize_mvars import compute_mvar_lifetime
from .utils import LoopingTransform
from .utils import TransformCtx


def _block_definitions_do_not_escape(index: Index, block: Block) -> bool:
    """
    Return True if every Var defined in `block.body` is only used within `block`.

    This is required for jump-threading: if we bypass `block`, we also skip its definitions.
    """
    block_instrs = set(block.contents)
    for instr in block.body:
        for v in instr.outputs:
            if any(index.get_parent_rec(use) not in block_instrs for use in index.vars[v].uses):
                return False
    return True


def _is_writeback(index: Index, block: Block, instr: BoundInstr) -> bool:
    """
    Return True if `instr` is `%s = %v` with `%v` read from `%s` in `block` (as
    writeback_mvar_use inserts), which leaves `%s` unchanged.
    """
    if not (w := instr.isinst(WriteMVar)):
        return False
    (val,) = w.inputs_
    if not isinstance(val, Var):
        return False
    r = index.vars[val].def_instr.isinst(ReadMVar)
    return r is not None and r.instr.s == w.instr.s and index.get_parent_rec(r) in block.body


def _mid_is_trivial(index: Index, mid: Block) -> bool:
    return all(i.is_side_effect_free() or _is_writeback(index, mid, i) for i in mid.body)


def _value_at_end_of(ctx: TransformCtx, pred: Block, mid: Block, v: Var) -> Value | None:
    """
    If `v` is read in `mid` from an MVar whose value at the end of `pred` always comes from the
    same `WriteMVar`, return the value written there.
    """
    index = get_index.call_cached(ctx)

    v_def = index.vars[v].def_instr
    if index.get_parent_rec(v_def) not in mid.body:
        return None
    if not (r := v_def.isinst(ReadMVar)):
        return None

    lifetime = compute_mvar_lifetime.call_cached(ctx, r.instr.s)
    info = lifetime.reachable.get(pred.end)
    if info is None or info.external_path:
        return None

    defs = info.possible_defs
    if len(defs) != 1:
        return None
    if not (w := defs[0].isinst(WriteMVar)):
        return None

    (written,) = w.inputs_
    return written


def _as_concrete_branch_end(block: Block) -> tuple[Var[bool], Label, Label] | None:
    """
    Return the `(cond, true_label, false_label)` triple if `block` ends in a `Branch` with
    concrete label targets.
    """
    end = block.end.isinst(Branch)
    if not end:
        return None

    cond, l_true, l_false = end.inputs_
    if not isinstance(cond, Var):
        return None
    if not isinstance(l_true, Label) or not isinstance(l_false, Label):
        return None

    return cond, l_true, l_false


def _threaded_target_for_edge(
    ctx: TransformCtx,
    *,
    edge_value: bool,
    pred_cond: Var[bool],
    pred: Block,
    mid: Block,
) -> Label | None:
    """
    If `mid` is a trivial block that branches on a value already known at the end of `pred`
    (for the `pred -> mid` edge), return the ultimate jump target label.
    """
    branch = _as_concrete_branch_end(mid)
    if branch is None:
        return None
    mid_cond, mid_t, mid_f = branch

    if mid_cond == pred_cond:
        return mid_t if edge_value else mid_f

    written = _value_at_end_of(ctx, pred, mid, mid_cond)
    if isinstance(written, bool):
        return mid_t if written else mid_f
    if written == pred_cond:
        return mid_t if edge_value else mid_f
    return None


def _thread_jump_edge(ctx: TransformCtx, pred: Block, pred_end: BoundInstr[Jump]) -> bool:
    """
    Thread `pred: ...; Jump L_mid` where `mid` jumps through an MVar `pred` set to a label:

      pred:
        ...; s = L; Jump L_mid
      mid:
        v = s; s = v; Jump v

    becomes `Jump L`: a static task's first step goes straight to the task's start, and a call
    to an empty subroutine straight to its return label.
    """
    f = ctx.frag
    index = get_index.call_cached(ctx)

    (mid_label,) = pred_end.inputs_
    if not isinstance(mid_label, Label) or mid_label not in f.blocks:
        return False
    mid = f.blocks[mid_label]

    if not (mid_end := mid.end.isinst(Jump)):
        return False
    (target,) = mid_end.inputs_
    if not isinstance(target, Var):
        return False
    if not _mid_is_trivial(index, mid) or not _block_definitions_do_not_escape(index, mid):
        return False

    threaded_target = _value_at_end_of(ctx, pred, mid, target)
    if not isinstance(threaded_target, Label) or threaded_target not in f.blocks:
        return False
    # Threading into another block that jumps through a variable could cycle (pred:
    # `s = L_b; t = L_a; Jump L_a`, with block a jumping through s and block b through t).
    if (t_end := f.blocks[threaded_target].end.isinst(Jump)) and isinstance(t_end.inputs_[0], Var):
        return False

    @f.replace_instr(pred_end)
    def _():
        return Jump().bind((), threaded_target)

    return True


@LoopingTransform
def thread_branch_edges(ctx: TransformCtx) -> bool:
    """
    Jump-threading for short-circuit lowering patterns.

    Detects patterns like:

      pred:
        ...; tmp = cond; Branch cond L_true L_mid
      mid:
        v = tmp; Branch v L_a L_b

    And replaces the pred->mid edge with pred->(L_a or L_b) based on which edge was taken.
    Also threads a jump into a jump through a known label (see _thread_jump_edge).
    """

    f = ctx.frag
    index = get_index.call_cached(ctx)

    for pred in f.blocks.values():
        if jump_end := pred.end.isinst(Jump):
            if _thread_jump_edge(ctx, pred, jump_end):
                return True
            continue

        if not (pred_end := pred.end.isinst(Branch)):
            continue

        pred_cond, l_true, l_false = pred_end.inputs_
        if not isinstance(pred_cond, Var):
            continue

        for edge_value, mid_label in ((True, l_true), (False, l_false)):
            if not isinstance(mid_label, Label):
                continue
            if mid_label not in f.blocks:
                continue

            mid = f.blocks[mid_label]
            if not _mid_is_trivial(index, mid):
                continue
            if not _block_definitions_do_not_escape(index, mid):
                continue

            threaded_target = _threaded_target_for_edge(
                ctx,
                edge_value=edge_value,
                pred_cond=pred_cond,
                pred=pred,
                mid=mid,
            )
            if threaded_target is None:
                continue

            new_l_true = threaded_target if edge_value else l_true
            new_l_false = l_false if edge_value else threaded_target
            if new_l_true == l_true and new_l_false == l_false:
                continue

            @f.replace_instr(pred_end)
            def _():
                return Branch().bind((), pred_cond, new_l_true, new_l_false)

            return True

    return False
