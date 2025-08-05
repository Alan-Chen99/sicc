from contextlib import contextmanager
from dataclasses import dataclass

import networkx as nx
from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.pretty import pretty_repr
from rich.text import Text

from .._core import FORMAT_ANNOTATE
from .._core import AlwaysUnpack
from .._core import BoundInstr
from .._core import MVar
from .._core import Var
from .._utils import cast_unchecked_val
from .._utils import mk_ordered_set
from .basic import MVarInfo
from .basic import VarInfo
from .basic import get_index
from .control_flow import CfgNode
from .control_flow import build_control_flow_graph
from .control_flow import external
from .optimize_mvars import compute_mvar_lifetime
from .utils import CachedFn
from .utils import TransformCtx

AnyVar = MVar | Var


@dataclass
class RegallocLifetimeRes:
    # live vars AFTER a instr
    live_vars: dict[CfgNode, OrderedSet[AnyVar]]

    def annotate(self, instr: BoundInstr) -> Text:
        if info := self.live_vars.get(instr):
            return Text(pretty_repr(list(info)), "ic10.comment")
        return Text()

    @contextmanager
    def with_anno(self):
        with FORMAT_ANNOTATE.bind(self.annotate):
            yield


@CachedFn
def compute_lifetimes_all(ctx: TransformCtx) -> RegallocLifetimeRes:
    index = get_index.call_cached(ctx, AlwaysUnpack())
    graph = build_control_flow_graph.call_cached(ctx, out_unpack=AlwaysUnpack())

    ans: dict[CfgNode, OrderedSet[AnyVar]] = {x: mk_ordered_set() for x in index.instrs_unpacked()}
    ans[external] = mk_ordered_set()

    def handle_one(info: VarInfo | MVarInfo, reachable: set[CfgNode]):
        for instr in info.defs:
            ans[instr].append(info.v)

        for instr in info.uses:
            if any(suc in reachable for suc in graph.successors(instr)):
                ans[instr].append(info.v)

        for instr in reachable:
            if instr not in info.uses:
                ans[instr].append(info.v)

    for mv in index.mvars.values():
        res = compute_mvar_lifetime(ctx, mv.v, AlwaysUnpack())
        handle_one(mv, set(res.reachable))

    for v in index.vars.values():
        hide_defs = cast_unchecked_val(graph)(
            nx.restricted_view(  # pyright: ignore[reportUnknownMemberType]
                graph, [v.def_instr], []
            ),
        )
        reachable: set[CfgNode] = set()
        for use in v.uses:
            reachable |= nx.ancestors(hide_defs, use)  # pyright: ignore[reportUnknownMemberType]

        handle_one(v, set(reachable))

    return RegallocLifetimeRes(ans)
