from .._core import Block
from .._core import WriteMVar
from .._instructions import Jump
from .._utils import is_eq_typed
from .basic import get_index
from .control_flow import build_control_flow_graph
from .control_flow import external
from .optimize_mvars import compute_mvar_lifetime
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
        if not (i.is_side_effect_free() or i.isinst(WriteMVar)):
            return False

    if len(b.body) == 0:
        return False

    # self looping
    if is_eq_typed(b_end.inputs_[0])(b.label):
        return False

    preds = list(graph.predecessors(b.label_instr))
    if len(preds) != 1 or external in preds:
        return False
    (pred_block,) = [x for x in f.blocks.values() if x.end == preds[0]]

    # handled in fuse
    if len(list(graph.successors(pred_block.end))) <= 1:
        return False

    for i in b.contents:
        if not (i := i.isinst(WriteMVar)):
            continue

        mv = i.instr.s
        if not index.mvars[mv].private:
            return False

        lifetime = compute_mvar_lifetime(ctx, mv)
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
