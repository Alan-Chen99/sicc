from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import cast

import networkx as nx
from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.pretty import pretty_repr
from rich.text import Text

from .._core import FORMAT_ANNOTATE
from .._core import FORMAT_VAL_FN
from .._core import AlwaysUnpack
from .._core import BoundInstr
from .._core import MVar
from .._core import NeverUnpack
from .._core import ReadMVar
from .._core import RegallocExtend
from .._core import RegallocSkip
from .._core import RegallocTie
from .._core import RegInfo
from .._core import Register
from .._core import Value
from .._core import Var
from .._core import WriteMVar
from .._diagnostic import mk_error
from .._instructions import Transmute
from .._tracing import mk_mvar
from .._tracing import mk_var
from .._utils import ByIdMixin
from .._utils import Cell
from .._utils import disjoint_union
from .._utils import get_id
from .._utils import in_typed
from ..config import verbose
from .basic import MVarInfo
from .basic import VarInfo
from .basic import get_index
from .basic import map_mvars
from .basic import map_vars
from .control_flow import CfgNode
from .label_contexts import CtxNode
from .label_contexts import build_label_context_graph
from .label_contexts import ctx_node_instr
from .optimize_mvars import MvarLifetimeRes
from .optimize_mvars import compute_mvar_lifetime
from .utils import CachedFn
from .utils import Transform
from .utils import TransformCtx

AnyVar = MVar | Var


@dataclass
class RegallocLifetimeRes:
    # live vars AFTER a instr
    lifetimes: dict[AnyVar, OrderedSet[CfgNode]]

    conflict_graph: "nx.Graph[AnyVar]"

    def annotate(self, instr: BoundInstr) -> Text:
        if vars := [x for x, y in self.lifetimes.items() if in_typed(instr, y)]:
            return Text(pretty_repr(vars, max_width=1000), "ic10.comment")
        return Text()

    @contextmanager
    def with_anno(self):
        with FORMAT_ANNOTATE.bind(self.annotate):
            yield


@CachedFn
def compute_lifetimes_all(ctx: TransformCtx) -> RegallocLifetimeRes:
    """
    Lifetimes are sets of nodes of the label context graph: a variable live after one call of a
    subroutine, or at one resume point of a static coroutine, is live only in the contexts that
    return there.
    """
    f = ctx.frag
    index = get_index.call_cached(ctx, AlwaysUnpack())
    index_nounpack = get_index.call_cached(ctx, NeverUnpack())
    graph = build_label_context_graph.call_cached(ctx)

    ctx_lifetimes: dict[AnyVar, OrderedSet[CtxNode]] = {}

    def handle_one(info: VarInfo | MVarInfo, hidden: set[BoundInstr]) -> Iterator[CtxNode]:
        # nodes from which a use is reachable without passing a node of `hidden`
        reachable: OrderedSet[CtxNode] = OrderedSet(graph.nodes_of(info.uses))
        work = list(reachable)
        while work:
            for p in graph.preds[work.pop()]:
                if p not in reachable and ctx_node_instr(p) not in hidden:
                    reachable.add(p)
                    work.append(p)

        yield from graph.nodes_of(info.defs)

        for instr in info.uses:
            extend = isinstance(info.v, Var) and RegallocExtend(info.v) in instr.regalloc_prefs()
            for n in graph.nodes.get(instr, []):
                if extend or any(suc in reachable for suc in graph.succs[n]):
                    yield n

        for n in reachable:
            if ctx_node_instr(n) not in info.uses:
                yield n

    mvar_lifetimes: dict[MVar, MvarLifetimeRes] = {}

    for mv in index.mvars.values():
        mvar_lifetimes[mv.v] = compute_mvar_lifetime.call_cached(ctx, mv.v, AlwaysUnpack())
        ctx_lifetimes[mv.v] = OrderedSet(handle_one(mv, set(mv.defs - mv.uses)))

    for v in index.vars.values():
        v_nounpack = index_nounpack.vars[v.v]
        if len(v_nounpack.uses) == 0 and in_typed(
            RegallocSkip(v.v), v_nounpack.def_instr.regalloc_prefs()
        ):
            continue

        # note: vars are never live at "external" due to their invariant
        ctx_lifetimes[v.v] = OrderedSet(handle_one(v, {v.def_instr}))

    lifetimes: dict[AnyVar, OrderedSet[CfgNode]] = {
        x: OrderedSet(ctx_node_instr(n) for n in ns) for x, ns in ctx_lifetimes.items()
    }

    live_at: dict[CtxNode, list[AnyVar]] = {}
    for x, ns in ctx_lifetimes.items():
        for n in ns:
            live_at.setdefault(n, []).append(x)
    live_together: set[tuple[AnyVar, AnyVar]] = set()
    for xs in live_at.values():
        for x in xs:
            for y in xs:
                if x < y:
                    live_together.add((x, y))

    def transmute_source(v: Var[Any]) -> Var[Any]:
        while (t := index.vars[v].def_instr.isinst(Transmute)) and isinstance(t.inputs_[0], Var):
            v = cast(Var[Any], t.inputs_[0])
        return v

    def copied_value(d: BoundInstr) -> tuple[MVar, Var[Any]] | None:
        """(mv, v) if d is `_ := v` with v read from mv"""
        if (w := d.isinst(WriteMVar)) and isinstance(v := w.inputs_[0], Var):
            if r := index.vars[v].def_instr.isinst(ReadMVar):
                return r.instr.s, cast(Var[Any], v)
        return None

    def changed_while_live(mv: MVar, v: AnyVar) -> bool:
        keep = preserving_writes(mv)
        return any(w not in keep and in_typed(w, lifetimes[v]) for w in index.mvars[mv].defs)

    preserving: dict[MVar, set[BoundInstr]] = {}

    def preserving_writes(mv: MVar) -> set[BoundInstr]:
        """
        Writes `mv := v` that keep the value of mv: v is read from mv, and no other write
        changes mv while v is live. (mvars are written back after each read)
        """
        if (ans := preserving.get(mv)) is not None:
            return ans
        cand: dict[BoundInstr, Var[Any]] = {}
        for d in index.mvars[mv].defs:
            if (c := copied_value(d)) and c[0] == mv:
                cand[d] = c[1]
        changed = True
        while changed:
            changed = False
            for d, v in list(cand.items()):
                if any(w not in cand and in_typed(w, lifetimes[v]) for w in index.mvars[mv].defs):
                    del cand[d]
                    changed = True
        preserving[mv] = set(cand)
        return preserving[mv]

    copies: dict[tuple[MVar, MVar], bool] = {}

    def copy_of(y: MVar, x: MVar) -> bool:
        if (ans := copies.get((y, x))) is None:
            ans = copies[(y, x)] = _copy_of(y, x)
        return ans

    def _copy_of(y: MVar, x: MVar) -> bool:
        """
        Whether y, wherever it is live, holds the current value of x: each write of y stores a
        value read from x, or keeps the value of y, with no write changing x since the read, and
        no write changes x while y is live. Arguments of a subroutine are copies of the caller's
        variables.
        """
        for d in index.mvars[y].defs:
            c = copied_value(d)
            if c is None or not (c[0] == x or d in preserving_writes(y)):
                return False
            if changed_while_live(x, c[1]):
                return False
        return not changed_while_live(x, y)

    def may_conflict(x: AnyVar, y: AnyVar) -> bool:
        if (x, y) not in live_together:
            return False
        # instructions where both are live in the same context
        together = OrderedSet(ctx_node_instr(n) for n in ctx_lifetimes[x] & ctx_lifetimes[y])
        conflicts = OrderedSet(c for c in lifetimes[x] if c in together)

        if isinstance(x, MVar) and isinstance(y, MVar):
            return not (copy_of(x, y) or copy_of(y, x))
        elif isinstance(x, Var) and isinstance(y, Var):
            # vars never change, and a transmute has the value of its input, so a var and its
            # transmutes can share a register
            return transmute_source(x) != transmute_source(y)
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
                if in_typed(mv_write, lifetimes[v]):
                    if not ((w := mv_write.isinst(WriteMVar)) and w.inputs_[0] == v):
                        return True
            return False

        # v is read from a copy of mv, or mv is a copy of what v is read from, and neither
        # changes while v is live?
        if (
            i
            and (copy_of(i.instr.s, mv) or copy_of(mv, i.instr.s))
            and not (changed_while_live(mv, v) or changed_while_live(i.instr.s, v))
        ):
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
                if not all(
                    (x_ := x.isinst(WriteMVar)) and x_.inputs_[0] == v for x in info.possible_defs
                ):
                    # mv maybe something other than v at c
                    return True

                # we want to say that mv has the same value as v at c
                # if that is not the case, we must have start -> v:=f(1) -> mv:=v -> v:=f(1) -> c
                # this would imply a path start -> v:=f(1) -> c, so c is "possible_undef"

                if info.external_path:
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

    ans = RegallocLifetimeRes(lifetimes=lifetimes, conflict_graph=conflict_graph)
    if verbose.value >= 1:
        with ans.with_anno():
            print(f.__rich__("live vars at end of each instruction:"))
    return ans


def _do_fuse(x: RegInfo, y: RegInfo) -> RegInfo:
    return RegInfo(
        preferred_reg=x.preferred_reg or y.preferred_reg,
        preferred_weight=x.preferred_weight + y.preferred_weight,
    )


counter = Cell(0)


@dataclass(eq=False)
class VarGroup(ByIdMixin):
    """a set of fused vars"""

    id: int
    name: Cell[str | None]
    vars: OrderedSet[AnyVar]
    reg: RegInfo

    @staticmethod
    def create(vars: Iterable[AnyVar], reg: RegInfo) -> VarGroup:
        return VarGroup(get_id(), Cell(None), OrderedSet(vars), reg)

    def get_name(self) -> str:
        if len(self.vars) == 1:
            return repr(self.vars[0])
        if self.name.value is None:
            ct = counter.value
            counter.value += 1
            self.name.value = f"g{ct}"
        return self.name.value

    def show(self) -> str:
        ans = ""
        if len(self.vars) > 1:
            ans += self.get_name()
        if self.reg.preferred_reg:
            ans += f":{self.reg.preferred_reg.value}"
        if ans:
            ans = f"[{ans}]"
        return ans


def _do_fuse_group(x: VarGroup, y: VarGroup) -> VarGroup:
    return VarGroup.create(disjoint_union(x.vars, y.vars), _do_fuse(x.reg, y.reg))


@dataclass
class RegallocFuseRes:
    groups: dict[AnyVar, VarGroup]

    group_conflict_graph: "nx.Graph[VarGroup]"

    def annotate(self, instr: BoundInstr) -> Text:
        if ties := _get_possible_fuse(instr):
            if all(self.groups[tie.v1] == self.groups[tie.v2] for tie in ties):
                return Text("fused", "ic10.comment")
            else:
                return Text("fuse failure", "ic10.comment")

        return Text()

    def format_val(self, v: Value | MVar) -> str:
        if isinstance(v, MVar | Var):
            if g := self.groups.get(v):
                return f"{v!r}{g.show()}"
            else:
                return f"{v!r}[skip]"
        return repr(v)

    @contextmanager
    def with_anno(self):
        with FORMAT_ANNOTATE.bind(self.annotate), FORMAT_VAL_FN.bind(self.format_val):
            yield


def _get_possible_fuse(instr: BoundInstr) -> list[RegallocTie]:
    return [x.normalize() for x in instr.regalloc_prefs() if isinstance(x, RegallocTie)]


@CachedFn
def regalloc_try_fuse(ctx: TransformCtx) -> RegallocFuseRes:
    """
    this does, in order:

    (1) for instrs of the form x:=y, try to fuse x and y if possible
    (2) fuse vars with the same preferred register;
        if there is conflict, drop the preference on the var with lower weight.
        this is done until for each reg, there is at most one var prefering that reg
    """
    f = ctx.frag
    index = get_index.call_cached(ctx, AlwaysUnpack())
    lifetime = compute_lifetimes_all.call_cached(ctx)

    # for each var, which group it is currently in
    # members in a group would point to the same VarGroup object
    groups: dict[AnyVar, VarGroup] = {x: VarGroup.create([x], x.reg) for x in lifetime.lifetimes}

    graph: nx.Graph[VarGroup] = nx.Graph()
    for n in groups.values():
        graph.add_node(n)
    for x, y in lifetime.conflict_graph.edges:
        graph.add_edge(groups[x], groups[y])

    def can_fuse(x: VarGroup, y: VarGroup) -> bool:
        if graph.has_edge(x, y):
            return False

        if x.reg.force_reg or y.reg.force_reg:
            raise NotImplementedError()

        x_pref = x.reg.preferred_reg
        y_pref = y.reg.preferred_reg

        if x_pref:
            if y_pref:
                return x_pref == y_pref
            else:
                pass
        else:
            if y_pref:
                x, y = y, x
                x_pref, y_pref = y_pref, x_pref
            else:
                return True

        # x has pref, y no pref
        # check if fusing would create a *new* conflict with another group that wants
        # the same preferred reg (which would then force one preference to be dropped)
        for z in graph.neighbors(y):
            if z.reg.preferred_reg == x_pref and not graph.has_edge(x, z):
                return False

        return True

    def perform_fuse(x: VarGroup, y: VarGroup) -> VarGroup:
        comb = _do_fuse_group(x, y)

        graph.add_node(comb)
        for n in list(graph.neighbors(x)) + list(graph.neighbors(y)):
            graph.add_edge(comb, n)

        graph.remove_node(x)
        graph.remove_node(y)

        for v, g in groups.items():
            if g == x or g == y:
                groups[v] = comb

        return comb

    ########################################
    # fuse trivial assignment x := y

    possible_fuses: dict[RegallocTie, int] = {}

    for instr in index.instrs_unpacked():
        for tie in _get_possible_fuse(instr):
            possible_fuses[tie] = possible_fuses.get(tie, 0) + (100 if tie.force else 1)

    possible_fuses_ = sorted(possible_fuses.items(), key=lambda item: item[1], reverse=True)
    for tie, _w in possible_fuses_:
        x = groups[tie.v1]
        y = groups[tie.v2]

        if x != y and can_fuse(x, y):
            perform_fuse(x, y)
        else:
            if tie.force:
                raise RuntimeError(
                    f"unable to satisfy requirement that {tie.v1} and {tie.v2} is allocated the same register"
                )

    ########################################
    # handle preferred_reg

    have_pref = OrderedSet(
        x.reg.preferred_reg for x in groups.values() if x.reg.preferred_reg is not None
    )
    for reg in have_pref:
        pref_groups = OrderedSet(g for g in groups.values() if g.reg.preferred_reg == reg)
        pref_groups = sorted(pref_groups, key=lambda g: g.reg.preferred_weight, reverse=True)

        cur = pref_groups[0]
        for other in pref_groups[1:]:
            if can_fuse(cur, other):
                cur = perform_fuse(cur, other)
            else:
                logging.info(
                    f"group {other.get_name()} cannot be assigned to its prefered reg {reg}"
                )
                # we can mutate this since the groups are created within this function
                other.reg = RegInfo()

    ans = RegallocFuseRes(groups, graph)

    if verbose.value >= 1:
        with ans.with_anno():
            print("regalloc fuse groups:", {g.get_name(): list(g.vars) for g in groups.values()})
            print(
                "with pref:",
                list(OrderedSet(g for g in groups.values() if g.reg.preferred_reg)),
            )
            print(f.__rich__("regalloc fuse result"))

    return ans


@dataclass
class RegAssignment:
    assignment: dict[Var | MVar, Register]


@Transform
def regalloc(ctx: TransformCtx) -> None:
    f = ctx.frag
    fuse_res = regalloc_try_fuse.call_cached(ctx)

    # free_regs = OrderedSet(OrderedSet(Register) - OrderedSet([Register.SP]))
    free_regs = OrderedSet(Register)

    ########################################
    # assign colors;
    # we already made sure there is at most on var prefering a reg in regalloc_try_fuse

    colors = cast(
        dict[VarGroup, int],
        nx.coloring.greedy_color(  # pyright: ignore[reportUnknownMemberType]
            fuse_res.group_conflict_graph, strategy="saturation_largest_first"
        ),
    )

    for c in colors.values():
        if c >= len(free_regs):
            err = mk_error("register spilling is not implemented")
            need = max(colors.values()) + 1
            err.note(f"{need} registers required; {len(free_regs)} is available")
            err.throw()

    ########################################
    # map colors to registers;
    # we already allocated ones that have pref.
    #
    # if something non-pref is colored the same as a pref,
    # they would use the pref register
    #
    # if two pref regs gets the same color,
    # then anything else can be alloctated with either one of the pref regs

    have_pref = OrderedSet(x for x in fuse_res.groups.values() if x.reg.preferred_reg is not None)

    color_map: dict[int, Register] = {}
    for g in have_pref:
        reg = g.reg.preferred_reg
        assert reg is not None
        assert reg not in color_map.values()
        color_map[colors[g]] = reg

    # give each remaining color a reg
    remain = OrderedSet(free_regs - OrderedSet(color_map.values()))

    for i in range(len(free_regs)):
        if i not in color_map:
            r = remain[0]
            remain.remove(r)
            color_map[i] = r

    ########################################
    # replace vars & mvars with a new assigned-reg var

    var_mapping: dict[Var, Var] = {}
    mvar_mapping: dict[MVar, MVar] = {}

    for v, g in fuse_res.groups.items():
        assert v.reg.allocated_reg is None

        if pref := g.reg.preferred_reg:
            reg = RegInfo(allocated_reg=pref)
        else:
            reg = RegInfo(allocated_reg=color_map[colors[g]])
        if isinstance(v, Var):
            var_mapping[v] = mk_var(v.type, reg=reg, debug=v.debug)
        else:
            mvar_mapping[v] = mk_mvar(v.type, reg=reg, debug=v.debug)

    map_vars(f, var_mapping)
    map_mvars(f, mvar_mapping)
