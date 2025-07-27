from dataclasses import dataclass

import networkx as nx

from .._core import Block
from .._core import BoundInstr
from .._core import Fragment
from .._core import Label
from .._core import MVar
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._tracing import internal_transform
from .._utils import Singleton
from .basic import BasicIndex
from .basic import get_basic_index


class External(Singleton):
    pass


# represents outside the fragment
external = External()

_NodeT = Var | MVar | Label | BoundInstr | External

JumpTarget = Label | External


@dataclass
class LabelProvenanceRes:
    index: BasicIndex  # reusable index since compute_label_provenance does not mutate
    instr_jumps: dict[BoundInstr, list[JumpTarget]]


@internal_transform
def compute_label_provenance(f: Fragment) -> LabelProvenanceRes:
    """does not mutate f"""
    # TODO: side effect (storing label on the stack)

    # note: it may be faster to compute SCC of the graph first and use it to do this "one-shot"
    #   which is asymptotically faster

    index = get_basic_index(f)

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
        for x in instr.ouputs:
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

        # if not l.internal:
        #     # we dont trace "external-listed" -> "var" from "external-listed"
        #     # because from the perspective of the fragment, "external-listed" is just "external"
        #     # so we add "external" -> "external-listed"
        #     G.add_edge(external, l.v)

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

    def handle_one(x: BoundInstr, ts: list[JumpTarget]) -> list[JumpTarget]:
        if x.does_jump():
            if external in ts:
                # not useful to track/output a [jump instr] -> external -> internal
                # as the internal is a possible entrypoint anyways
                ts = [t for t in ts if (isinstance(t, External) or index.labels[t].private)]
            return ts
        return []

    res = {x: handle_one(x, y) for x, y in reaches_label.items() if isinstance(x, BoundInstr)}

    # print(index)

    # print({x: y for x, y in res.items() if x.does_jump()})
    # print("res", res)
    return LabelProvenanceRes(index, res)


CfgNode = BoundInstr | External


@internal_transform
def build_instr_flow_graph(
    f: Fragment,
) -> tuple["nx.DiGraph[CfgNode]", LabelProvenanceRes]:
    res = compute_label_provenance(f)
    index = res.index

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

        for l in index.labels.values():
            if l.internal and not l.private:
                G.add_edge(external, l.def_instr)

    return G, res


@internal_transform
def remove_unreachable_code(f: Fragment) -> None:
    G, _res = build_instr_flow_graph(f)

    live_instrs = nx.descendants(G, external)  # pyright: ignore[reportUnknownMemberType]

    def process_block():
        for b in f.blocks.values():
            ans: list[BoundInstr] = []
            for x in b.contents:
                if x in live_instrs:
                    ans.append(x)
                else:
                    x.debug.mark_unused()

            if len(ans) > 0:
                yield Block(ans, b.debug)

    f.blocks = {x.label: x for x in process_block()}
