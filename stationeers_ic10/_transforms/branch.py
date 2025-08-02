from .._core import Var
from .._diagnostic import add_debug_info
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
            v, l_t, l_f = instr.inputs_
            if isinstance(v, Var):
                v = v.check_type(bool)
                if def_instr := index.vars[v].def_instr.isinst(PredicateBase):

                    @f.replace_instr(instr)
                    def _():
                        assert def_instr is not None
                        new_v = mk_var(bool)
                        with add_debug_info(def_instr.debug):
                            return PredBranch(def_instr.instr).bind(
                                (new_v,), l_t, l_f, *def_instr.inputs
                            )

                    return True

    return False
