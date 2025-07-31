from dataclasses import dataclass
from dataclasses import field

import networkx as nx
from ordered_set import OrderedSet
from rich import print as print  # autoflake: skip

from .._core import Block
from .._core import BoundInstr
from .._core import EffectBase
from .._core import EffectMvar
from .._core import Label
from .._core import MapInstrsRes
from .._core import MVar
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._core import known_distinct
from .._instructions import EmitLabel
from .._instructions import Jump
from .._tracing import ck_val
from .._tracing import mk_internal_label
from .._tracing import mk_mvar
from .._tracing import mk_var
from .._utils import Cell
from .utils import CachedFn
from .utils import LoopingTransform
from .utils import Transform
from .utils import TransformCtx


@dataclass
class VarInfo:
    v: Var
    defs: list[BoundInstr] = field(default_factory=lambda: [])
    uses: list[BoundInstr] = field(default_factory=lambda: [])

    @property
    def def_instr(self) -> BoundInstr:
        assert len(self.defs) == 1
        return self.defs[0]


@dataclass
class MVarInfo:
    v: MVar
    defs: list[BoundInstr] = field(default_factory=lambda: [])
    uses: list[BoundInstr] = field(default_factory=lambda: [])
    private: bool = False


@dataclass
class LabelInfo:
    v: Label
    defs: list[BoundInstr] = field(default_factory=lambda: [])
    uses: list[BoundInstr] = field(default_factory=lambda: [])

    # private implies internal
    private: bool = False

    @property
    def internal(self) -> bool:
        return len(self.defs) > 0

    @property
    def def_instr(self) -> BoundInstr:
        assert len(self.defs) == 1
        return self.defs[0]


@dataclass
class EffectInfo:
    loc: EffectBase

    reads_instrs: list[BoundInstr] = field(default_factory=lambda: [])
    writes_instrs: list[BoundInstr] = field(default_factory=lambda: [])


@dataclass
class Index:
    vars: dict[Var, VarInfo]

    mvars: dict[MVar, MVarInfo]

    labels: dict[Label, LabelInfo]

    effects: dict[EffectBase, EffectInfo]

    effect_conflicts: "nx.Graph[EffectBase]"


@CachedFn
def get_index(ctx: TransformCtx) -> Index:
    f = ctx.frag
    res_vars: dict[Var, VarInfo] = {}
    # res_mvars: dict[MVar, MVarInfo] = {}
    res_labels: dict[Label, LabelInfo] = {}
    res_effects: dict[EffectBase, EffectInfo] = {}

    def get_var(v: Var) -> VarInfo:
        assert isinstance(v, Var)
        return res_vars.setdefault(v, VarInfo(v, [], []))

    def get_label(v: Label) -> LabelInfo:
        assert isinstance(v, Label)
        return res_labels.setdefault(v, LabelInfo(v))

    def get_effect(loc: EffectBase) -> EffectInfo:
        assert isinstance(loc, EffectBase)
        return res_effects.setdefault(loc, EffectInfo(loc))

    @f.map_instrs
    def _(instr: BoundInstr):
        for v in instr.inputs:
            ck_val(v)
            if isinstance(v, Var):
                get_var(v).uses.append(instr)
            if isinstance(v, Label):
                get_label(v).uses.append(instr)

        for v in instr.outputs:
            ck_val(v)
            get_var(v).defs.append(instr)

        if i := instr.isinst(EmitLabel):
            (arg,) = i.inputs_
            assert not isinstance(arg, Var)
            l = get_label(arg)
            l.defs.append(instr)

        for loc in instr.reads():
            get_effect(loc).reads_instrs.append(instr)

        for loc in instr.writes():
            get_effect(loc).writes_instrs.append(instr)

    res_mvars = {
        x.loc.s: MVarInfo(x.loc.s, defs=x.writes_instrs, uses=x.reads_instrs)
        for x in res_effects.values()
        if isinstance(x.loc, EffectMvar)
    }

    # note: we dont add private_mvar members that are no longer relavant
    for x in res_mvars.values():
        if x.v in f.scope.private_mvars:
            x.private = True

    for x in res_labels.values():
        if x.v in f.scope.private_labels:
            x.private = True

    # FIXME: move this, this function should not mutate scope
    f.scope.private_mvars = OrderedSet(x for x in f.scope.private_mvars if x in res_mvars)
    f.scope.private_labels = OrderedSet(x for x in f.scope.private_labels if x in res_labels)
    f.scope.vars = OrderedSet(x for x in f.scope.vars if x in res_vars)

    # sanity checks
    for x in res_vars.values():
        if len(x.defs) != 1:
            x.v.debug.error(f"{x.v} defined {len(x.defs)} times")

    for l in res_labels.values():
        assert len(l.defs) <= 1
        if l.private and len(l.uses) > 0:
            if len(l.defs) == 0:
                l.v.debug.error(f"private label {l} never defined")

    effect_list = list(res_effects)
    effect_conflicts: "nx.Graph[EffectBase]" = nx.Graph(nodes=effect_list)

    for i, x in enumerate(effect_list):
        for y in effect_list[i:]:
            if not known_distinct(x, y):
                effect_conflicts.add_edge(x, y)

    return Index(res_vars, res_mvars, res_labels, res_effects, effect_conflicts)


@Transform
def split_blocks(ctx: TransformCtx) -> None:
    """
    split blocks based on "continues"

    ensures that there is no labels in the middle of any block
    """
    f = ctx.frag

    def handle_block(b: Block):
        cur: list[BoundInstr] = []

        def get() -> Block:
            nonlocal cur
            ans = Block(cur, b.debug)
            ans.basic_check()
            cur = []
            return ans

        for x in b.contents:
            if len(cur) == 0 and not x.isinst(EmitLabel):
                cur.append(EmitLabel().bind((), mk_internal_label("_split_blocks_dead")))

            if len(cur) > 0 and (x_ := x.isinst(EmitLabel)):
                (l,) = x_.inputs_
                cur.append(Jump().bind((), l))
                yield get()

            cur.append(x)
            if not x.instr.continues:
                ans = Block(cur, b.debug)
                ans.basic_check()
                yield ans
                cur = []

        assert len(cur) == 0

    blocks: list[Block] = []

    for b in f.blocks.values():
        blocks += list(handle_block(b))

    f.blocks = {b.label: b for b in blocks}


@LoopingTransform
def remove_unused_side_effect_free(ctx: TransformCtx) -> bool:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    ans = Cell(False)

    @f.map_instrs
    def _(instr: BoundInstr):
        if instr.isinst(EmitLabel) or len(instr.jumps_to()) > 0 or not instr.instr.continues:
            return None
        if not instr.is_side_effect_free():
            return None
        for x in instr.outputs:
            if len(index.vars[x].uses) > 0:
                return None
        ans.value = True
        return ()

    return ans.value


@Transform
def rename_private_vars(ctx: TransformCtx) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    new_vars = {x.v: mk_var(x.v.type, debug=x.v.debug) for x in index.vars.values()}

    @f.map_instrs
    def _(instr: BoundInstr) -> MapInstrsRes:
        return instr.instr.bind(  # pyright: ignore
            tuple(new_vars[x] for x in instr.outputs),
            *(new_vars[x] if isinstance(x, Var) else x for x in instr.inputs),
        )


@Transform
def rename_private_labels(ctx: TransformCtx) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    new_labels = {
        x.v: mk_internal_label(x.v.id) if x.private else x.v for x in index.labels.values()
    }

    @f.map_instrs
    def _(instr: BoundInstr) -> MapInstrsRes:
        return instr.instr.bind(  # pyright: ignore
            instr.outputs, *(new_labels[x] if isinstance(x, Label) else x for x in instr.inputs)
        )


@Transform
def rename_private_mvars(ctx: TransformCtx) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    new_mvars = {
        x.v: mk_mvar(x.v.type, debug=x.v.debug) if x.private else x.v for x in index.mvars.values()
    }

    @f.map_instrs
    def _(instr: BoundInstr) -> MapInstrsRes:
        if any(isinstance(x, EffectMvar) for x in instr.reads()):
            if i := instr.isinst(ReadMVar):
                return ReadMVar(new_mvars[i.instr.s]).bind(i.outputs_, *i.inputs_)
            raise TypeError(instr)

        if any(isinstance(x, EffectMvar) for x in instr.writes()):
            if i := instr.isinst(WriteMVar):
                return WriteMVar(new_mvars[i.instr.s]).bind(i.outputs_, *i.inputs_)
            raise TypeError(instr)


@Transform
def mark_all_private_except(ctx: TransformCtx, labels: list[Label]) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for l in index.mvars:
        f.scope.private_mvars.add(l)
    for l in index.labels:
        if l not in labels:
            f.scope.private_labels.add(l)
