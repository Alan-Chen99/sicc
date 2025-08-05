from .._core import Var
from .._instructions import Branch
from .._instructions import PredBranch
from .._instructions import PredicateBase
from .._tracing import mk_var
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def inline_pred_to_branch(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for b in f.blocks.values():
        if instr := b.end.isinst(Branch):
            v, _l_t, _l_f = instr.inputs_
            if isinstance(v, Var):
                v = v.check_type(bool)
                if def_instr := index.vars[v].def_instr.isinst(PredicateBase):

                    @f.replace_instr(instr)
                    def _():
                        assert def_instr is not None
                        new_v = mk_var(bool)
                        ans = PredBranch.from_parts(def_instr, instr)
                        return ans.sub_val(v, new_v, outputs=True)

                    return True

    return False
