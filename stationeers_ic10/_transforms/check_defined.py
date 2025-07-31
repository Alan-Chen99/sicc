import networkx as nx

from .._utils import cast_unchecked_val
from .basic import get_index
from .control_flow import build_control_flow_graph
from .control_flow import external
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
    index = get_index.call_cached(ctx)
    graph = build_control_flow_graph.call_cached(ctx)

    for v in index.vars.values():
        graph_ = cast_unchecked_val(graph)(
            nx.restricted_view(graph, {v.def_instr}, [])  # pyright: ignore[reportUnknownMemberType]
        )
        reach = nx.descendants(graph_, external)  # pyright: ignore[reportUnknownMemberType]
        for u in v.uses:
            if u in reach:
                err = u.debug.error(
                    f"variable {v.v} ({v.v.debug.describe}) may not be initialized",
                )
                err.note("defined here:", v.v.debug)
                err.note("initialized here, but may occur after use:", v.def_instr.debug)


@CachedFn
def check_mvars_defined(ctx: TransformCtx) -> None:
    index = get_index.call_cached(ctx)
    graph = build_control_flow_graph.call_cached(ctx)

    for v in index.mvars.values():
        if not v.private:
            continue
        graph_ = cast_unchecked_val(graph)(
            nx.restricted_view(graph, set(v.defs), [])  # pyright: ignore[reportUnknownMemberType]
        )

        reach = nx.descendants(graph_, external)  # pyright: ignore[reportUnknownMemberType]
        for u in v.uses:
            if u in reach:
                err = u.debug.error(
                    f"mvar {v.v} ({v.v.type}) may not be initialized",
                )
                err.note("defined here:", v.v.debug)
                for d in v.defs:
                    err.note("initialized here, but may occur after use:", d.debug)
