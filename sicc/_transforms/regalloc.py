from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import networkx as nx
from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.pretty import pretty_repr
from rich.text import Text

from .._core import FORMAT_ANNOTATE
from .._core import AlwaysUnpack
from .._core import BoundInstr
from .._core import MVar
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._instructions import Move
from .._utils import cast_unchecked_val
from .._utils import in_typed
from .._utils import isinst
from .._utils import mk_ordered_set
from .basic import MVarInfo
from .basic import VarInfo
from .basic import get_index
from .control_flow import CfgNode
from .control_flow import build_control_flow_graph
from .optimize_mvars import MvarLifetimeRes
from .optimize_mvars import compute_mvar_lifetime
from .utils import CachedFn
from .utils import TransformCtx

AnyVar = MVar | Var


@dataclass
class RegallocLifetimeRes:
    # live vars AFTER a instr
    lifetimes: dict[AnyVar, OrderedSet[CfgNode]]

    conflict_graph: "nx.Graph[AnyVar]"

    def annotate(self, instr: BoundInstr) -> Text:
        if vars := [x for x, y in self.lifetimes.items() if in_typed(instr, y)]:
            return Text(pretty_repr(vars), "ic10.comment")
        return Text()

    @contextmanager
    def with_anno(self):
        with FORMAT_ANNOTATE.bind(self.annotate):
            yield


@CachedFn
def compute_lifetimes_all(ctx: TransformCtx) -> RegallocLifetimeRes:
    index = get_index.call_cached(ctx, AlwaysUnpack())
    graph = build_control_flow_graph.call_cached(ctx, out_unpack=AlwaysUnpack())

    lifetimes: dict[AnyVar, OrderedSet[CfgNode]] = {
        x: mk_ordered_set() for x in list(index.mvars) + list(index.vars)
    }

    def handle_one(info: VarInfo | MVarInfo, reachable: set[CfgNode]) -> Iterator[CfgNode]:
        for instr in info.defs:
            yield instr

        for instr in info.uses:
            if any(suc in reachable for suc in graph.successors(instr)):
                yield instr

        for instr in reachable:
            if instr not in info.uses:
                yield instr

    mvar_lifetimes: dict[MVar, MvarLifetimeRes] = {}

    for mv in index.mvars.values():
        res = compute_mvar_lifetime(ctx, mv.v, AlwaysUnpack())
        mvar_lifetimes[mv.v] = res
        lifetimes[mv.v] = OrderedSet(handle_one(mv, set(res.reachable)))

    for v in index.vars.values():
        hide_defs = cast_unchecked_val(graph)(
            nx.restricted_view(  # pyright: ignore[reportUnknownMemberType]
                graph, [v.def_instr], []
            ),
        )
        reachable: set[CfgNode] = set()
        for use in v.uses:
            reachable |= nx.ancestors(hide_defs, use)  # pyright: ignore[reportUnknownMemberType]

        # note: vars are never live at "external" due to their invariant
        lifetimes[v.v] = OrderedSet(handle_one(v, set(reachable)))

    def may_conflict(x: AnyVar, y: AnyVar) -> bool:
        conflicts = lifetimes[x] & lifetimes[y]
        if len(conflicts) == 0:
            return False

        if isinstance(x, MVar) and isinstance(y, MVar):
            return True
        elif isinstance(x, Var) and isinstance(y, Var):
            return True
        elif isinstance(x, MVar) and isinstance(y, Var):
            mv, v = x, y
        elif isinstance(x, Var) and isinstance(y, MVar):
            mv, v = y, x
        else:
            assert False

        # v is read from mv?
        if (i := index.vars[v].def_instr.isinst(ReadMVar)) and i.instr.s == mv:
            # we want to know whether mv will get overwritten within the lifetime of v
            for mv_write in index.mvars[mv].defs:
                assert (mv_write := mv_write.isinst(WriteMVar))
                if in_typed(mv_write, lifetimes[v]) and mv_write.inputs_[0] != v:
                    return True
            return False

        # mv contains a read of v?
        for c in conflicts:
            reachable = mvar_lifetimes[mv].reachable
            if c not in reachable:
                # c is a mv:=blah
                assert isinstance(c, BoundInstr)
                assert (c_ := c.isinst(WriteMVar))
                assert c_.instr.s == mv

                if c_.inputs_ == (v,):
                    # mv := v instr
                    continue
                else:
                    return True

            else:
                info = reachable[c]
                if not all(x.inputs_[0] == v for x in info.possible_defs):
                    # mv maybe something other than v at c
                    return True

                # we want to say that mv has the same value as v at c
                # if that is not the case, we must have start -> v:=f(1) -> mv:=v -> v:=f(1) -> c
                # this would imply a path start -> v:=f(1) -> c, so c is "possible_undef"

                if info.possible_undef:
                    return False

                # here we can conclude that mv and v does not conflict at c

        # print(f"{mv} and {v} not conflicting: {mv} have a read of {v}")
        # so mv contain a read of v
        return False

    conflict_graph: nx.Graph[AnyVar] = nx.Graph()
    for x in lifetimes:
        for y in lifetimes:
            if x < y and may_conflict(x, y):
                conflict_graph.add_edge(x, y)

    return RegallocLifetimeRes(lifetimes=lifetimes, conflict_graph=conflict_graph)


@dataclass
class RegallocFuseRes:
    groups: dict[AnyVar, AnyVar]
    groups_rev: dict[AnyVar, OrderedSet[AnyVar]]

    collapsed_conflict_graph: "nx.Graph[AnyVar]"

    def annotate(self, instr: BoundInstr) -> Text:
        if edge := _get_possible_fuse(instr):
            x, y = edge
            x, y = self.groups[x], self.groups[y]
            if x == y:
                return Text("fused", "ic10.comment")
            else:
                return Text("fuse failure", "ic10.comment")

        return Text()

    @contextmanager
    def with_anno(self):
        with FORMAT_ANNOTATE.bind(self.annotate):
            yield


def _get_possible_fuse(instr: BoundInstr) -> tuple[AnyVar, AnyVar] | None:
    if i := instr.isinst(Move):
        (x,) = i.inputs_
        (y,) = i.outputs_
        if not isinst(x, Var):
            return None

    elif i := instr.isinst(ReadMVar):
        x = i.instr.s
        (y,) = i.outputs_

    elif i := instr.isinst(WriteMVar):
        (x,) = i.inputs_
        y = i.instr.s
        if not isinst(x, Var):
            return None

    else:
        return None

    x, y = sorted([x, y])
    return x, y


@CachedFn
def regalloc_try_fuse(ctx: TransformCtx) -> RegallocFuseRes:
    index = get_index.call_cached(ctx, AlwaysUnpack())
    lifetime = compute_lifetimes_all.call_cached(ctx)

    possible_fuses: dict[tuple[AnyVar, AnyVar], int] = {}

    for instr in index.instrs_unpacked():
        if edge := _get_possible_fuse(instr):
            possible_fuses[edge] = possible_fuses.get(edge, 0) + 1

    # print("possible_fuses", possible_fuses)

    possible_fuses_ = sorted(possible_fuses.items(), key=lambda item: item[1], reverse=True)

    groups: dict[AnyVar, AnyVar] = {x: x for x in lifetime.lifetimes}
    graph = lifetime.conflict_graph.copy()

    for (x, y), _w in possible_fuses_:
        x = groups[x]
        y = groups[y]

        if graph.has_edge(x, y):
            continue

        # combine x and y

        for n in list(graph.neighbors(y)):
            graph.add_edge(x, n)

        graph.remove_node(y)

        groups = {var: x if prev_group == y else prev_group for var, prev_group in groups.items()}

    groups_rev: dict[AnyVar, OrderedSet[AnyVar]] = {x: mk_ordered_set() for x in groups.values()}
    for v, g in groups.items():
        groups_rev[g].add(v)

    return RegallocFuseRes(groups, groups_rev, graph)
