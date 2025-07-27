from .._core import Fragment
from .basic import split_blocks
from .check_defined import check_vars_defined
from .label_provenance import remove_unreachable_code
from .remove_trivial_vars import remove_trivial_vars_mvars


def normalize(f: Fragment) -> Fragment:
    split_blocks(f)
    remove_unreachable_code(f)
    check_vars_defined(f)
    remove_trivial_vars_mvars(f)

    # compute_label_provenance(f)
    # print(f)
    # index = get_basic_index(f)
    # print(index)
    # print(f)
    return f
