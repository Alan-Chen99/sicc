from .._core import Fragment
from .basic import remove_unused_side_effect_free
from .basic import split_blocks
from .branch import inline_pred_to_branch
from .check_defined import check_vars_defined
from .label_provenance import remove_unreachable_code
from .remove_trivial_vars import remove_trivial_mvars
from .remove_trivial_vars import remove_trivial_vars
from .utils import run_phases


def normalize(f: Fragment) -> Fragment:
    _changed = run_phases(
        f,
        split_blocks,
        remove_unreachable_code,
        check_vars_defined,
        remove_trivial_vars,
        remove_trivial_mvars,
        inline_pred_to_branch,
        remove_unused_side_effect_free,
    )

    # compute_label_provenance(f)
    # print(f)
    # index = get_basic_index(f)
    # print(index)
    # print(f)
    return f
