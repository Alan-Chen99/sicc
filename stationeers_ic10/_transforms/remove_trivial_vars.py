from .._core import BoundInstr
from .._instructions import Move
from .basic import get_basic_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def remove_trivial_vars_(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_basic_index.call_cached(ctx)

    for v in index.vars.values():
        if def_instr := v.def_instr.isinst(Move):
            (source,) = def_instr.inputs_

            @f.map_instrs
            def _(instr: BoundInstr):
                if instr == def_instr:
                    return ()
                if instr in v.uses:
                    return instr.sub_val(v.v, source, inputs=True)

            # if isinstance(source, Var):
            #     source.debug.fuse(v.v.debug)

            return True

    return False
