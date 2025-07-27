import networkx as nx

from .._core import Fragment
from .._tracing import internal_transform
from .._utils import cast_unchecked_val
from .label_provenance import LabelProvenanceRes
from .label_provenance import build_instr_flow_graph
from .label_provenance import external


@internal_transform
def check_vars_defined(f: Fragment) -> LabelProvenanceRes:
    """
    check that vars are always defined before use, using the graph.

    this does not mutate f

    this property is required for some passes (such as [%vy := %vx] ==> make theses vars the same)
    it is NOT sufficient for %vy to be guaranteed initialized at runtime; it must be a property of the graph

    returns cached result from "build_instr_flow_graph"
    """
    graph, res = build_instr_flow_graph(f)
    index = res.index

    for v in index.vars.values():
        graph_ = cast_unchecked_val(graph)(
            nx.restricted_view(graph, {v.def_instr}, [])  # pyright: ignore[reportUnknownMemberType]
        )
        reach = nx.descendants(graph_, external)  # pyright: ignore[reportUnknownMemberType]
        for u in v.uses:
            if u in reach:
                err = u.debug.error(
                    f"variable {v.v} ({v.v.type}) may not be initialized",
                )
                err.note("defined here:", v.v.debug)
                err.note("initialized here, but may occur after use:", v.def_instr.debug)

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

    return res
