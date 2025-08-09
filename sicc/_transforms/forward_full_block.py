from .._core import AlwaysUnpack
from .._core import Block
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._instructions import Bundle
from .._instructions import Jump
from .._utils import is_eq_typed
from .basic import get_index
from .control_flow import build_control_flow_graph
from .control_flow import external
from .optimize_mvars import compute_mvar_lifetime
from .optimize_mvars import support_mvar_analysis
from .utils import LoopingTransform
from .utils import TransformCtx


def _try_forward_once(ctx: TransformCtx, b: Block) -> bool:
    """
    try to move a post-branch block entirely to before-branch. most notably:

    if_(...):
        fn(x+1)

    into

    y = x + 1
    if_(...):
        fn(y)
    """
    f = ctx.frag
    graph = build_control_flow_graph.call_cached(ctx)
    index = get_index.call_cached(ctx)

    if not (b_end := b.end.isinst(Jump)):
        return False

    for i in b.contents:
        # TODO: is_side_effect_free would be correct here if the op is garanteed
        # to not fail. does failure (read from non-exist device for ex) cause problem?
        safe_side_effect_free = i.is_pure() or i.isinst(ReadMVar)
        if not (safe_side_effect_free or i.isinst(WriteMVar)):
            return False

    if len(b.body) == 0:
        return False

    # self looping
    if is_eq_typed(b_end.inputs_[0])(b.label):
        return False

    preds = list(graph.predecessors(b.label_instr))
    if len(preds) != 1 or external in preds:
        return False
    pred_blocks = [x for x in f.blocks.values() if x.end == preds[0]]
    if len(pred_blocks) == 0:
        # comes from probably a condition jump;
        # not implemented
        return False
    (pred_block,) = pred_blocks

    if (pred_end := pred_block.end.isinst(Jump)) and isinstance(pred_end.inputs_[0], Var):
        # previous block looks like a function return
        # the optimization would be correct;
        # but it may not be a good idea to move stuff into it, so we dont do this opt for now
        return False

    # handled in fuse
    if len(list(graph.successors(pred_block.end))) <= 1:
        return False

    for i in b.contents:
        if not (i := i.isinst(WriteMVar)):
            continue

        mv = i.instr.s
        if not index.mvars[mv].private:
            return False

        if pred_block.end.isinst(Bundle):
            # TODO
            return False

        if not support_mvar_analysis(ctx, mv, AlwaysUnpack()):
            return False
        lifetime = compute_mvar_lifetime(ctx, mv, AlwaysUnpack())
        if pred_block.end in lifetime.reachable:
            return False

    pred_block.contents = pred_block.contents[:-1] + b.body + [pred_block.end]
    b.contents = [b.label_instr, b.end]

    return True


@LoopingTransform
def forward_remove_full_block(ctx: TransformCtx) -> bool:
    # note: this pass moves stuff up the dominator tree
    # so not possible for infinite loop

    f = ctx.frag
    for b in f.blocks.values():
        if _try_forward_once(ctx, b):
            return True

    return False
