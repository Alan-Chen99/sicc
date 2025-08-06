from typing import Iterator

from .._core import Block
from .._core import BoundInstr
from .._core import Label
from .._core import Var
from .._diagnostic import add_debug_info
from .._instructions import CondJump
from .._instructions import EndPlaceholder
from .._instructions import Jump
from .._instructions import PredBranch
from .._instructions import PredCondJump
from .basic import get_index
from .control_flow import External
from .control_flow import build_control_flow_graph
from .utils import LoopingTransform
from .utils import Transform
from .utils import TransformCtx


@LoopingTransform
def remove_trivial_blocks(ctx: TransformCtx) -> bool:
    """
    remove block that is just a jump
    """
    f = ctx.frag
    index = get_index.call_cached(ctx)
    for b in f.blocks.values():
        if (
            #
            len(b.contents) == 2
            and index.labels[b.label].private
            and (instr := b.end.isinst(Jump))
            and not isinstance(instr.inputs_[0], Var)
            and (instr.inputs_[0] != b.label)
        ):
            del f.blocks[b.label]
            (rep,) = instr.inputs_

            @f.map_instrs
            def _(instr: BoundInstr):
                return instr.sub_val(b.label, rep, inputs=True, strict=False)

            return True

    return False


def _fuse_blocks_impl(ctx: TransformCtx, trivial_only: bool, efficient_only: bool) -> bool:
    if trivial_only:
        assert efficient_only

    f = ctx.frag
    index = get_index.call_cached(ctx)
    graph = build_control_flow_graph.call_cached(ctx)

    def attempt_fuse(cur: Block, target: Label) -> bool:
        # in the middle of some other block
        if target not in f.blocks:
            return False

        # it loops back to itself
        if target == b.label:
            return False

        # this is probably the main "entrypoint" so dont fuse to it
        if not index.labels[target].private:
            return False

        target_block = f.blocks[target]

        if trivial_only:
            # jump [label] should be the only use
            if len(index.labels[target].uses) > 1:
                return False
        pred = list(graph.predecessors(target_block.contents[0]))
        if efficient_only:
            for x in pred:
                assert not isinstance(x, External)
                if x != cur.end and not x.continues:
                    return False

        # do the fuse

        del f.blocks[target]

        # TODO: make use of "target_block.label.implicit"
        # and keep user created labels
        # currently we cant keep the non-implicit label since we have a "split_blocks"

        # if len(pred) == 1 and target_block.label.implicit:
        if len(pred) == 1:
            append = target_block.contents[1:]
        else:
            assert not trivial_only
            append = target_block.contents

        if b.end.isinst(Jump):
            keep = b.contents[:-1]
        elif instr := b.end.isinst(PredBranch):
            pred, br = instr.unpack()
            predvar, t_l, f_l = br.inputs_

            with add_debug_info(instr.debug):
                if t_l == target:
                    rep = PredCondJump.from_parts(
                        pred.instr.negate(pred), CondJump().bind((), predvar, f_l)
                    )
                else:
                    assert f_l == target
                    rep = PredCondJump.from_parts(pred, CondJump().bind((), predvar, t_l))
            keep = b.contents[:-1] + [rep]
        else:
            assert False

        b.contents = keep + append
        return True

    for b in f.blocks.values():
        if instr := b.end.isinst(Jump):
            (target,) = instr.inputs_
            if isinstance(target, Var):
                continue

            if attempt_fuse(b, target):
                return True

        if not trivial_only and (instr := b.end.isinst(PredBranch)):
            _pred, br = instr.unpack()
            _predvar, t_l, f_l = br.inputs_
            if isinstance(t_l, Label) and attempt_fuse(b, t_l):
                return True
            if isinstance(f_l, Label) and attempt_fuse(b, f_l):
                return True

        # targets = res.instr_jumps[b.end]

        # if len(targets) != 1:
        #     continue
        # (target,) = targets
        # if isinstance(target, External):
        #     continue
        # if not index.labels[target].private:
        #     continue

    return False


@LoopingTransform
def fuse_blocks_trivial_jumps(ctx: TransformCtx) -> bool:
    """
    fuse a jump to a private label, if that label is only used for this jump
    """
    return _fuse_blocks_impl(ctx, trivial_only=True, efficient_only=True)


@LoopingTransform
def fuse_blocks_all(ctx: TransformCtx) -> bool:
    if _fuse_blocks_impl(ctx, trivial_only=False, efficient_only=True):
        return True
    return _fuse_blocks_impl(ctx, trivial_only=False, efficient_only=False)


@Transform
def force_fuse_into_one(ctx: TransformCtx, start: Label) -> None:
    f = ctx.frag
    if len(f.blocks) == 1:
        assert f.blocks[start].end.isinst(EndPlaceholder)
        return

    start_block = f.blocks[start]
    (end_block,) = (b for b in f.blocks.values() if b.end.isinst(EndPlaceholder))
    assert start_block is not end_block
    other_blocks = [b for b in f.blocks.values() if b is not start_block and b is not end_block]

    def gen() -> Iterator[BoundInstr]:
        yield from start_block.contents
        for b in other_blocks:
            yield from b.contents
        yield from end_block.contents

    ans = Block(list(gen()), start.debug)

    f.blocks = {start: ans}
