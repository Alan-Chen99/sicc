from dataclasses import dataclass
from typing import Iterator

import networkx as nx
from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip
from rich.text import Text

from .._core import AlwaysUnpack
from .._core import Block
from .._core import BoundInstr
from .._core import EffectBase
from .._core import Label
from .._core import NeverUnpack
from .._core import Undef
from .._core import UnpackPolicy
from .._core import Var
from .._instructions import EmitLabel
from .._instructions import Jump
from .._instructions import UnreachableChecked
from .._utils import Cell
from .._utils import Singleton
from .basic import get_index
from .utils import CachedFn
from .utils import LoopingTransform
from .utils import Transform
from .utils import TransformCtx
from .utils import frag_is_global


class External(Singleton):
    pass


# represents outside the fragment
external = External()

_NodeT = Var | Label | BoundInstr | External | EffectBase

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
        if jumps := self.instr_jumps.get(instr):
            start, end = "{", "}"
        elif instr in self.reaches_label:
            start, end = "[", "]"
            jumps = self.reaches_label[instr]
        else:
            return Text("...")

        ans = Text(start, "ic10.comment")
        ans += Text(", ", "ic10.comment").join(
            Text(repr(x), "ic10.label" if x in self.leaked else "ic10.label_private") for x in jumps
        )
        ans += Text(end, "ic10.comment")

        return ans


@CachedFn
def compute_label_provenance(
    ctx: TransformCtx,
    *,
    out_unpack: UnpackPolicy = NeverUnpack(),
    analysis_unpack: UnpackPolicy = AlwaysUnpack(),
) -> LabelProvenanceRes:
    """
    this computes possible values varaibles with type Label can take.
    this is required to build a control flow graph.

    does not mutate f
    """
    # note: it may be faster to compute SCC of the graph first and use it to do this "one-shot"
    #   which is asymptotically faster

    index = get_index.call_cached(ctx, analysis_unpack)
    out_index = get_index.call_cached(ctx, out_unpack)

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

    for instr in index.instrs_unpacked():
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

    for ef in index.effects.values():
        for instr in ef.reads_instrs:
            G.add_edge(ef.loc, instr)
        for instr in ef.writes_instrs:
            G.add_edge(instr, ef.loc)

        if not frag_is_global.value:
            G.add_edge(external, ef.loc)
            G.add_edge(ef.loc, external)

    for x, y in index.effect_conflicts.edges:
        G.add_edge(x, y)
        G.add_edge(y, x)

    for l in index.labels.values():
        if not l.private:
            # if var may hold "external": "external" -> "var"
            # then "var" may also hold "l"
            # so we add "l" -> "external"
            # print(("adding:", external, l.v))
            G.add_edge(l.v, external)

    # print(f)

    reaches_label: dict[_NodeT, list[JumpTarget]] = {x: [] for x in G.nodes}

    for l in labels:
        res = nx.descendants(G, l)  # pyright: ignore[reportUnknownMemberType]
        for x in res:
            reaches_label[x].append(l)

    # print("reaches_label", reaches_label)

    def handle_one(instr: BoundInstr) -> Iterator[JumpTarget]:
        for child in instr.unpack_rec_or_self(analysis_unpack):
            for x in child.jumps_to():
                if isinstance(x, Label):
                    yield preprocess(x)
                if isinstance(x, Var):
                    yield from reaches_label[x]

    res = {instr: sorted(set(handle_one(instr))) for instr in out_index.instrs_unpacked()}

    # print(index)

    def leaked():
        for x in reaches_label[external]:
            assert not isinstance(x, External)
            yield x

    # print({x: y for x, y in res.items() if x.does_jump()})
    # print("res", res)
    res = LabelProvenanceRes(
        instr_jumps=res,
        leaked=OrderedSet(leaked()),
        reaches_label=reaches_label,
    )

    # with FORMAT_ANNOTATE.bind(res.annotate):
    #     print(f)

    return res


CfgNode = BoundInstr | External


@CachedFn
def build_control_flow_graph(
    ctx: TransformCtx, *, out_unpack: UnpackPolicy = NeverUnpack()
) -> "nx.DiGraph[CfgNode]":
    f = ctx.frag
    index = get_index.call_cached(ctx, out_unpack)
    res = compute_label_provenance.call_cached(ctx, out_unpack=out_unpack)

    # # note: this creates extra labels
    # split_blocks(f)

    G: nx.DiGraph[BoundInstr | External] = nx.DiGraph()
    G.add_node(external)

    for b in f.blocks.values():
        contents = [x for instr in b.contents for x in instr.unpack_rec_or_self(out_unpack)]
        for x, y in zip(contents[:-1], contents[1:]):
            if x.continues:
                G.add_edge(x, y)

        for x in contents:
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

    dead_labels: list[Label] = []

    for b in f.blocks.values():
        for x in b.contents:
            if not x in live_instrs and (i := x.isinst(EmitLabel)):
                (label,) = i.inputs_
                assert isinstance(label, Label)
                dead_labels.append(label)

    def process_block():
        for b in f.blocks.values():
            ans: list[BoundInstr] = []
            for x in b.contents:
                if x in live_instrs:
                    for arg in x.inputs:
                        if arg in dead_labels:
                            x = x.sub_val(arg, Undef(Label), inputs=True)
                    ans.append(x)

            if len(ans) > 0:
                yield Block(ans, b.debug)

    f.blocks = {x.label: x for x in process_block()}


@LoopingTransform
def handle_deterministic_jump(ctx: TransformCtx) -> bool:
    f"""
    replace a jump %var with only one possible target,
    or a branch [...] %var label where var is always same as label

    this mostly comes from subr calls that is only called once
    """
    f = ctx.frag
    res = compute_label_provenance.call_cached(ctx)

    changed = Cell(False)

    @f.map_instrs
    def _(instr: BoundInstr):
        if not instr.is_pure():
            return None
        if instr.continues:
            return None

        targets = res.instr_jumps[instr]
        if len(targets) == 0 and instr.isinst(Jump):
            instr.debug.warn("no possible target for jump")
            changed.value = True
            return UnreachableChecked().bind(())

        if len(targets) != 1:
            return None
        (target,) = targets
        if isinstance(target, External):
            return None
        if (i := instr.isinst(Jump)) and i.inputs_[0] == target:
            return None

        changed.value = True
        return Jump().bind((), target)

    return changed.value
