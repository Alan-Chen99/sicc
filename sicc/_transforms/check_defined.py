import networkx as nx
from rich.text import Text

from .._core import FORMAT_ANNOTATE
from .._core import AlwaysUnpack
from .._diagnostic import mk_warn
from .._instructions import AsmBlock
from .._utils import cast_unchecked_val
from ..config import verbose
from .basic import get_index
from .control_flow import build_control_flow_graph
from .control_flow import compute_label_provenance
from .control_flow import external
from .optimize_mvars import compute_mvar_lifetime
from .optimize_mvars import support_mvar_analysis
from .utils import CachedFn
from .utils import TransformCtx


@CachedFn
def check_vars_defined(ctx: TransformCtx) -> None:
    """
    check that vars are always defined before use, using the graph.

    this does not mutate f

    this property is required for some passes (such as [%vy := %vx] ==> make theses vars the same)
    it is NOT sufficient for %vy to be guaranteed initialized at runtime; it must be a property of the graph

    returns cached result from "build_instr_flow_graph"
    """
    f = ctx.frag
    index = get_index.call_cached(ctx, unpack=AlwaysUnpack())
    label_prov = compute_label_provenance.call_cached(ctx, out_unpack=AlwaysUnpack())
    graph = build_control_flow_graph.call_cached(ctx, out_unpack=AlwaysUnpack())

    for v in index.vars.values():
        graph_ = cast_unchecked_val(graph)(
            nx.restricted_view(graph, {v.def_instr}, [])  # pyright: ignore[reportUnknownMemberType]
        )
        reach = nx.descendants(graph_, external)  # pyright: ignore[reportUnknownMemberType]
        for u in v.uses:
            if u in reach:
                err = u.debug.error(
                    f"variable {v.v} ({v.v.debug.describe}) does not dominiate its use",
                )
                err.note("this is an internal error, not a user error")
                err.note("in instruction", u)
                err.note("defined here:", v.v.debug)
                err.note("initialized here:", v.def_instr.debug)
                with FORMAT_ANNOTATE.bind(label_prov.annotate):
                    err.note("fragment:", f.__rich__())
                err.throw()


@CachedFn
def check_mvars_defined(ctx: TransformCtx) -> None:
    index = get_index.call_cached(ctx, AlwaysUnpack())

    for v in index.mvars.values():
        if not support_mvar_analysis(ctx, v.v, AlwaysUnpack()):
            continue
        res = compute_mvar_lifetime(ctx, v.v, AlwaysUnpack())

        undef_uses = [
            use for use in v.uses if res.reachable[use].external_path and not use.isinst(AsmBlock)
        ]
        if len(undef_uses) > 0 and verbose.value >= 1:
            err = mk_warn(f"mvar {v.v} ({v.v.type.__name__}) can not be proved to be initialized")
            err.note("lifetime analysis is currently limited, so this may not be an error")

            for i, u in enumerate(undef_uses):
                err.add(
                    Text(f"Possible uninitialized use [{i+1}]:", "bold"),
                    u.debug,
                    "",
                    "corresponding instruction:",
                    u,
                )
            err.note("defined here:", v.v.debug)
            for d in v.defs:
                err.note("assigned here, but may occur after use:", d.debug)
