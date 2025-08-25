from rich import print as print  # autoflake: skip

from .._core import BoundInstr
from .._instructions import Move
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def remove_trivial_vars_(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for v in index.vars.values():
        if def_instr := v.def_instr.isinst(Move):
            (source,) = def_instr.inputs_

            @f.map_instrs
            def _(instr: BoundInstr):
                if instr == def_instr:
                    return ()
                if instr in v.uses:
                    instr.debug.fuse_must_use(def_instr.debug)
                    instr.debug.fuse_must_use(v.v.debug)
                    return instr.sub_val(v.v, source, inputs=True)

            return True

    return False
