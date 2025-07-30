from dataclasses import dataclass
from typing import Iterator

import networkx as nx
from ordered_set import OrderedSet
from rich.text import Text

from .._core import Block
from .._core import BoundInstr
from .._core import Label
from .._core import MVar
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._instructions import Jump
from .._utils import Singleton
from .basic import get_basic_index
from .utils import CachedFn
from .utils import LoopingTransform
from .utils import Transform
from .utils import TransformCtx


class External(Singleton):
    pass


# represents outside the fragment
external = External()

_NodeT = Var | MVar | Label | BoundInstr | External

JumpTarget = Label | External


@dataclass
class LabelProvenanceRes:
    instr_jumps: dict[BoundInstr, list[JumpTarget]]

    #: set of internal labels accessible from outside
    #: either bc it is public, or bc it is assigned to a varaible readable outside
    leaked: OrderedSet[Label]

    #: raw outputs
    reaches_label: dict[_NodeT, list[JumpTarget]]

    def annotate(self, instr: BoundInstr) -> Text:
        # jumps = self.instr_jumps[instr]
        jumps = self.reaches_label[instr]

        ans = Text("{", "ic10.comment")
        ans += Text(", ", "ic10.comment").join(
            Text(repr(x), "ic10.label" if x in self.leaked else "ic10.label_private") for x in jumps
        )
        ans += Text("}", "ic10.comment")

        return ans


@CachedFn
def compute_label_provenance(ctx: TransformCtx) -> LabelProvenanceRes:
    """
    this computes possible values varaibles with type Label can take.
    this is required to build a control flow graph.

    does not mutate f
    """
    # TODO: side effect (storing label on the stack)

    # note: it may be faster to compute SCC of the graph first and use it to do this "one-shot"
    #   which is asymptotically faster

    f = ctx.frag
    index = get_basic_index.call_cached(ctx)

    G: nx.DiGraph[_NodeT] = nx.DiGraph()

    def preprocess(l: Label) -> JumpTarget:
        # jumping at a particular label outside frag is not meanningful
        # we turn (public, external) label into just "external" instead
        info = index.labels[l]
        if not info.internal:
            return external
        return l

    # jump statements may jump to one of these
    labels = [x.v for x in index.labels.values() if x.internal] + [external]
    for x in labels:
        G.add_node(x)

    @f.map_instrs
    def _(instr: BoundInstr):
        # ensure all instr is added as node even if no operands
        G.add_node(instr)

        # normal instrs
        for x in instr.inputs:
            if isinstance(x, Label):
                G.add_edge(preprocess(x), instr)
            if isinstance(x, Var):
                G.add_edge(x, instr)
        for x in instr.outputs:
            G.add_edge(instr, x)

        # mvars
        if i := instr.isinst(ReadMVar):
            G.add_edge(i.instr.s, instr)
        if i := instr.isinst(WriteMVar):
            G.add_edge(instr, i.instr.s)

        # TODO: things with write effect?
        #

    for l in index.labels.values():
        if not l.private:
            # if var may hold "external": "external" -> "var"
            # then "var" may also hold "l"
            # so we add "l" -> "external"
            # print(("adding:", external, l.v))
            G.add_edge(l.v, external)

    for var in index.mvars.values():
        if not var.private:
            G.add_edge(external, var.v)
            G.add_edge(var.v, external)

    # print(f)

    reaches_label: dict[_NodeT, list[JumpTarget]] = {x: [] for x in G.nodes}

    for l in labels:
        res = nx.descendants(G, l)  # pyright: ignore[reportUnknownMemberType]
        for x in res:
            reaches_label[x].append(l)

    # print("reaches_label", reaches_label)

    def handle_one(instr: BoundInstr) -> Iterator[JumpTarget]:
        if instr.does_jump():
            for x in instr.inputs:
                if isinstance(x, Label):
                    yield preprocess(x)
                if isinstance(x, Var) and x.type == Label:
                    yield from reaches_label[x]

    res = {instr: sorted(set(handle_one(instr))) for instr in f.all_instrs()}

    # print(index)

    def leaked():
        for x in reaches_label[external]:
            assert not isinstance(x, External)
            yield x

    # print({x: y for x, y in res.items() if x.does_jump()})
    # print("res", res)
    return LabelProvenanceRes(
        instr_jumps=res,
        leaked=OrderedSet(leaked()),
        reaches_label=reaches_label,
    )


CfgNode = BoundInstr | External


@CachedFn
def build_control_flow_graph(ctx: TransformCtx) -> "nx.DiGraph[CfgNode]":
    f = ctx.frag
    index = get_basic_index.call_cached(ctx)
    res = compute_label_provenance.call_cached(ctx)

    # # note: this creates extra labels
    # split_blocks(f)

    G: nx.DiGraph[BoundInstr | External] = nx.DiGraph()
    G.add_node(external)

    for b in f.blocks.values():
        for x, y in zip(b.contents[:-1], b.contents[1:]):
            if x.instr.continues:
                G.add_edge(x, y)

        for x in b.contents:
            for t in res.instr_jumps[x]:
                if isinstance(t, External):
                    G.add_edge(x, external)
                else:
                    G.add_edge(x, index.labels[t].def_instr)

        # for l in index.labels.values():
        #     if l.internal and not l.private:
        #         G.add_edge(external, l.def_instr)

    for l in res.leaked:
        G.add_edge(external, index.labels[l].def_instr)

    return G


@Transform
def remove_unreachable_code(ctx: TransformCtx) -> None:
    f = ctx.frag
    G = build_control_flow_graph.call_cached(ctx)

    live_instrs = nx.descendants(G, external)  # pyright: ignore[reportUnknownMemberType]

    def process_block():
        for b in f.blocks.values():
            ans: list[BoundInstr] = []
            for x in b.contents:
                if x in live_instrs:
                    ans.append(x)

            if len(ans) > 0:
                yield Block(ans, b.debug)

    f.blocks = {x.label: x for x in process_block()}


@LoopingTransform
def handle_deterministic_var_jump(ctx: TransformCtx) -> bool:
    f"""
    replace a jump %var with only one possible target,
    or a branch [...] %var label where var is always same as label

    this mostly comes from subr calls that is only called once
    """
    f = ctx.frag
    res = compute_label_provenance.call_cached(ctx)

    @f.map_instrs
    def _(instr: BoundInstr):
        targets = res.instr_jumps[instr]
        if len(targets) != 1:
            return None
        (target,) = targets
        if isinstance(target, External):
            return None
        assert instr.is_pure()

        return Jump().bind((), target)

    return False
