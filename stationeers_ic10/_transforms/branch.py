from .._core import Var
from .._instructions import Branch
from .._instructions import PredicateBase
from .._instructions import PredVar
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def inline_pred_to_branch(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for b in f.blocks.values():
        if (instr := b.end.isinst(Branch)) and isinstance(instr.instr.base, PredVar):
            l_t, l_f, v = instr.inputs_
            if isinstance(v, Var):
                v = v.check_type(bool)
                if def_instr := index.vars[v].def_instr.isinst(PredicateBase):

                    @f.replace_instr(instr)
                    def _():
                        assert def_instr is not None
                        return Branch(def_instr.instr).bind((), l_t, l_f, *def_instr.inputs)

                    return True

    return False
