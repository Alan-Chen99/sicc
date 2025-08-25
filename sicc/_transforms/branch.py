from .._core import Var
from .._diagnostic import add_debug_info
from .._instructions import Branch
from .._instructions import Jump
from .._instructions import Not
from .._instructions import PredBranch
from .._instructions import PredicateBase
from .._instructions import PredVar
from .._tracing import mk_var
from .basic import get_index
from .utils import LoopingTransform
from .utils import TransformCtx


@LoopingTransform
def handle_const_or_not_branch(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for b in f.blocks.values():
        if instr := b.end.isinst(Branch):
            v, l_t, l_f = instr.inputs_

            if isinstance(v, bool):

                @f.replace_instr(instr)
                def _():
                    return Jump().bind((), l_t if v else l_f)

                return True

            elif isinstance(v, Var) and (not_instr := index.vars[v].def_instr.isinst(Not)):
                (arg,) = not_instr.inputs_

                @f.replace_instr(instr)
                def _():
                    with add_debug_info(not_instr.debug):
                        return Branch().bind((), arg, l_f, l_t)

                return True

    return False


@LoopingTransform
def inline_pred_to_branch(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for b in f.blocks.values():
        if instr := b.end.isinst(Branch):
            v, l_t, l_f = instr.inputs_

            if isinstance(v, Var):
                if def_instr := index.vars[v].def_instr.isinst(PredicateBase):

                    @f.replace_instr(instr)
                    def _():
                        assert def_instr is not None
                        new_v = mk_var(bool)
                        ans = PredBranch.from_parts(def_instr, instr)
                        return ans.sub_val(v, new_v, outputs=True)

                    return True

            @f.replace_instr(instr)
            def _():
                new_v = mk_var(bool)
                return PredBranch.from_parts(
                    PredVar().bind((new_v,), v), Branch().bind((), new_v, l_t, l_f)
                )

            return True

    return False
