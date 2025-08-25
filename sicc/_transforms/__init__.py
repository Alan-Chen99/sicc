import logging
from typing import Callable

from rich import print as print  # autoflake: skip

from .._core import FORMAT_ANNOTATE
from .._core import Fragment
from ..config import verbose
from .arith import opt_identity_arith
from .basic import remove_unused_side_effect_free
from .basic import rename_private_labels
from .basic import rename_private_mvars
from .basic import rename_private_vars
from .basic import split_blocks
from .branch import handle_const_or_not_branch
from .branch import inline_pred_to_branch
from .check_defined import check_mvars_defined
from .check_defined import check_vars_defined
from .common_sub_elim import common_sub_elim
from .consteval import opt_consteval
from .control_flow import compute_label_provenance
from .control_flow import handle_deterministic_jump
from .control_flow import remove_unreachable_code
from .forward_full_block import forward_remove_full_block
from .fuse_blocks import fuse_blocks_all
from .fuse_blocks import fuse_blocks_trivial_jumps
from .fuse_blocks import remove_trivial_blocks
from .link_bundles import pack_call
from .link_bundles import pack_cond_call
from .lower import lower_instrs
from .offset_arith import opt_offset_arith
from .optimize_mvars import elim_mvars_read_writes
from .optimize_mvars import writeback_mvar_use
from .predicates import handle_nan_eq
from .regalloc import regalloc
from .remove_trivial_vars import remove_trivial_vars_
from .stack import stack_chain_to_push_pop
from .utils import frag_is_global
from .utils import run_phases

FRAG_OPTS: list[Callable[[Fragment], bool | None]] = [
    check_vars_defined,
    split_blocks,
    #
    remove_unreachable_code,
    # remove_trivial_mvars,
    remove_trivial_vars_,
    remove_trivial_blocks,
    #
    opt_consteval,
    opt_identity_arith,
    #
    remove_unused_side_effect_free,
    handle_deterministic_jump,
    fuse_blocks_trivial_jumps,
    handle_const_or_not_branch,
    #
    elim_mvars_read_writes,
    forward_remove_full_block,
    #
    opt_offset_arith,
    common_sub_elim,
]

GLOBAL_OPTS: list[Callable[[Fragment], bool | None]] = FRAG_OPTS + [
    handle_nan_eq,
    #
    inline_pred_to_branch,
    pack_cond_call,
    pack_call,
    stack_chain_to_push_pop,
]


def optimize_frag(f: Fragment) -> None:
    rename_private_vars(f)
    rename_private_labels(f)
    rename_private_mvars(f)
    run_phases(f, *FRAG_OPTS)


def global_checks(f: Fragment):
    with frag_is_global.bind(True):
        check_vars_defined(f)
        if verbose.value >= 1:
            with FORMAT_ANNOTATE.bind(compute_label_provenance(f).annotate):
                print(
                    f.__rich__("{possible jump targets} and [possible values Label vars can take]")
                )

        check_mvars_defined(f)


def global_opts(f: Fragment):
    with frag_is_global.bind(True):
        optimize_frag(f)
        writeback_mvar_use(f)
        logging.info("opts after writeback_mvar_use")
        optimize_frag(f)
        logging.info("running GLOBAL_OPTS")
        run_phases(f, *GLOBAL_OPTS)

        fuse_blocks_all(f, efficient_only=True)
        if verbose.value >= 1:
            print(f.__rich__("after fuse blocks with efficient_only"))
        fuse_blocks_all(f)


def regalloc_and_lower(f: Fragment):
    # fuse_blocks_all(f)
    regalloc(f)
    lower_instrs(f)
