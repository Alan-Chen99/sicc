"""
A control flow graph whose nodes are (instruction, values of label variables).

`j %ra` may go to any label %ra can hold (see compute_label_provenance). In the plain control flow
graph all these paths merge: a variable live after one call of a subroutine is live in the
subroutine and back from every other call of it, and a variable live at one resume point of a
static coroutine is live at every suspend point. Here each instruction is split by the labels the
label variables still read later are known to hold, so `j %ra` reached with %ra known to be L
goes only to L.
"""

from dataclasses import dataclass
from typing import Iterable
from typing import Iterator
from typing import cast

import networkx as nx

from .._core import AlwaysUnpack
from .._core import BoundInstr
from .._core import Label
from .._core import MVar
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._instructions import MoveBase
from .._utils import cast_unchecked_val
from .basic import get_index
from .control_flow import CfgNode
from .control_flow import External
from .control_flow import JumpTarget
from .control_flow import build_control_flow_graph
from .control_flow import compute_label_provenance
from .control_flow import external
from .optimize_mvars import compute_mvar_lifetime
from .utils import CachedFn
from .utils import LoopingTransform
from .utils import TransformCtx

type LabelVar = Var[Label] | MVar[Label]
#: the known values of label variables that may still be read
type Env = frozenset[tuple[LabelVar, Label]]
type CtxNode = tuple[BoundInstr, Env] | External


@dataclass
class LabelContextGraph:
    succs: dict[CtxNode, list[CtxNode]]
    preds: dict[CtxNode, list[CtxNode]]
    #: the nodes of each instruction reachable from the program start
    nodes: dict[BoundInstr, list[CtxNode]]

    def nodes_of(self, instrs: Iterable[BoundInstr]) -> Iterator[CtxNode]:
        for instr in instrs:
            yield from self.nodes.get(instr, [])


def ctx_node_instr(n: CtxNode) -> CfgNode:
    return n if isinstance(n, External) else n[0]


@CachedFn
def build_label_context_graph(ctx: TransformCtx) -> LabelContextGraph:
    f = ctx.frag
    index = get_index.call_cached(ctx, AlwaysUnpack())
    prov = compute_label_provenance.call_cached(ctx, out_unpack=AlwaysUnpack())
    cfg = build_control_flow_graph.call_cached(ctx, out_unpack=AlwaysUnpack())

    fallthrough: dict[BoundInstr, BoundInstr] = {}
    for b in f.blocks.values():
        contents = [x for instr in b.contents for x in instr.unpack_rec_or_self(AlwaysUnpack())]
        for x, y in zip(contents[:-1], contents[1:]):
            if x.continues:
                fallthrough[x] = y

    ########################################
    # label variables that may be read after each instruction; values of the others are dropped,
    # so an instruction gets one node per combination of values that can still matter

    live_after: dict[BoundInstr, set[LabelVar]] = {}

    def mark_live(v: LabelVar, live_before: set[CfgNode]) -> None:
        for x in live_before:
            for p in cfg.predecessors(x):
                if isinstance(p, BoundInstr):
                    live_after.setdefault(p, set()).add(v)

    for vi in index.vars.values():
        if vi.v.type is not Label:
            continue
        v = cast(Var[Label], vi.v)
        hide_def = cast_unchecked_val(cfg)(
            nx.restricted_view(cfg, [vi.def_instr], []),  # pyright: ignore[reportUnknownMemberType]
        )
        live_before: set[CfgNode] = set(vi.uses)
        for use in vi.uses:
            live_before |= nx.ancestors(hide_def, use)  # pyright: ignore[reportUnknownMemberType]
        mark_live(v, live_before)

    # MVars defined by something other than a WriteMVar get an unknown value there
    label_mvar_defs: dict[BoundInstr, list[MVar[Label]]] = {}
    for mvi in index.mvars.values():
        if mvi.v.type is not Label:
            continue
        mv = cast(MVar[Label], mvi.v)
        mark_live(mv, set(compute_mvar_lifetime.call_cached(ctx, mv, AlwaysUnpack()).reachable))
        for d in mvi.defs:
            label_mvar_defs.setdefault(d, []).append(mv)

    ########################################
    # transfer function and jump targets

    def value(x: object, env: dict[LabelVar, Label]) -> Label | None:
        if isinstance(x, Label):
            return x
        if isinstance(x, Var):
            return env.get(cast(Var[Label], x))
        return None

    def run(instr: BoundInstr, before: dict[LabelVar, Label]) -> Env:
        after = dict(before)

        for mv in label_mvar_defs.get(instr, []):
            val = value(w.inputs_[0], before) if (w := instr.isinst(WriteMVar)) else None
            if val is None:
                after.pop(mv, None)
            else:
                after[mv] = val

        for out in instr.outputs:
            if out.type is not Label:
                continue
            out_ = cast(Var[Label], out)
            if m := instr.isinst(MoveBase):
                val = value(m.inputs_[0], before)
            elif r := instr.isinst(ReadMVar):
                val = before.get(cast(MVar[Label], r.instr.s))
            else:
                val = None
            if val is None:
                after.pop(out_, None)
            else:
                after[out_] = val

        keep = live_after.get(instr, set())
        return frozenset((v, l) for v, l in after.items() if v in keep)

    def preprocess(l: Label) -> JumpTarget:
        return l if index.labels[l].internal else external

    def targets(instr: BoundInstr, env: dict[LabelVar, Label]) -> Iterator[JumpTarget]:
        for t in instr.jumps_to():
            if isinstance(t, Label):
                yield preprocess(t)
            elif isinstance(t, Var):
                if (l := env.get(t)) is not None:
                    target = preprocess(l)
                    assert target in prov.reaches_label[t]
                    yield target
                else:
                    yield from prov.reaches_label[t]

    ########################################
    # nodes reachable from the program start

    succs: dict[CtxNode, list[CtxNode]] = {}
    nodes: dict[BoundInstr, list[CtxNode]] = {}

    entry_env: Env = frozenset()
    succs[external] = [(x, entry_env) for x in cfg.successors(external) if isinstance(x, BoundInstr)]
    seen: set[CtxNode] = {external, *succs[external]}
    work: list[CtxNode] = list(succs[external])

    while work:
        n = work.pop()
        assert not isinstance(n, External)
        instr, env_in = n
        nodes.setdefault(instr, []).append(n)

        before = dict(env_in)
        env_out = run(instr, before)

        out: list[CtxNode] = []
        if (nxt := fallthrough.get(instr)) is not None:
            out.append((nxt, env_out))
        for t in targets(instr, before):
            out.append(external if isinstance(t, External) else (index.labels[t].def_instr, env_out))
        out = list(dict.fromkeys(out))

        succs[n] = out
        for m in out:
            if m not in seen:
                seen.add(m)
                work.append(m)

    preds: dict[CtxNode, list[CtxNode]] = {n: [] for n in succs}
    for n, ss in succs.items():
        for s in ss:
            preds[s].append(n)

    return LabelContextGraph(succs=succs, preds=preds, nodes=nodes)


@LoopingTransform
def handle_context_deterministic_jump(ctx: TransformCtx) -> bool:
    """
    Replace %var in `j %var`, or a branch to %var, by the label %var holds if it is the same in
    every context the instruction is reached in. handle_deterministic_jump replaces only a %var
    that can hold a single label at all; a static coroutine's suspend jumps through a variable set
    by whichever step is running, which holds one label per task.
    """
    f = ctx.frag
    prov = compute_label_provenance.call_cached(ctx)

    def targets(instr: BoundInstr) -> list[Var[Label]]:
        return [t for t in instr.jumps_to() if isinstance(t, Var) and len(prov.reaches_label[t]) > 1]

    if not any(targets(x) for b in f.blocks.values() for x in b.contents):
        return False

    graph = build_label_context_graph.call_cached(ctx)
    changed = False

    @f.map_instrs
    def _(instr: BoundInstr):
        nonlocal changed
        for t in targets(instr):
            values = {
                None if isinstance(n, External) else dict(n[1]).get(t)
                for n in graph.nodes.get(instr, [])
            }
            if len(values) == 1 and (label := values.pop()) is not None:
                instr = instr.sub_val(t, label, inputs=True)
                changed = True
        return instr

    return changed
