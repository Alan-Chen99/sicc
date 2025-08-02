from dataclasses import dataclass
from dataclasses import field
from weakref import WeakSet

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
from .._core import NeverUnpack
from .._core import ReadMVar
from .._core import UnpackPolicy
from .._core import Var
from .._core import WriteMVar
from .._core import known_distinct
from .._instructions import EmitLabel
from .._instructions import EndPlaceholder
from .._instructions import Isolate
from .._instructions import Jump
from .._tracing import ck_val
from .._tracing import mk_internal_label
from .._tracing import mk_mvar
from .._tracing import mk_var
from .._utils import Cell
from .._utils import in_typed
from .._utils import mk_ordered_set
from .utils import CachedFn
from .utils import LoopingTransform
from .utils import Transform
from .utils import TransformCtx

# note: use ordered set since code uses len() to check #
# previously instrs emiiting label are pushed twice
# so we change everything to OrderSet to prevent future bugs


@dataclass
class VarInfo:
    v: Var
    defs: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)
    uses: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)

    @property
    def def_instr(self) -> BoundInstr:
        assert len(self.defs) == 1
        return self.defs[0]


@dataclass
class MVarInfo:
    v: MVar
    defs: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)
    uses: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)
    private: bool = False


@dataclass
class LabelInfo:
    v: Label
    # comes from "defines_labels" method on InstrBase
    defs: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)
    # any uses that is not in "defs", so disjoint with defs
    uses: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)

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

    #: instruction that reads from loc
    reads_instrs: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)

    #: instruction that writes into loc
    writes_instrs: OrderedSet[BoundInstr] = field(default_factory=mk_ordered_set)


@dataclass
class InstrInfo:
    i: BoundInstr
    # this bundle directly contains...
    children: list[BoundInstr] | None
    # direct parent
    parent: BoundInstr | None


@dataclass
class Index:
    vars: dict[Var, VarInfo]

    mvars: dict[MVar, MVarInfo]

    labels: dict[Label, LabelInfo]

    effects: dict[EffectBase, EffectInfo]
    effect_conflicts: "nx.Graph[EffectBase]"

    instrs: dict[BoundInstr, InstrInfo]

    def instrs_unpacked(self) -> list[BoundInstr]:
        return [i.i for i in self.instrs.values() if i.children is None]


@CachedFn
def get_index(ctx: TransformCtx, unpack: UnpackPolicy = NeverUnpack()) -> Index:
    f = ctx.frag
    res_vars: dict[Var, VarInfo] = {}
    # res_mvars: dict[MVar, MVarInfo] = {}
    res_labels: dict[Label, LabelInfo] = {}
    res_effects: dict[EffectBase, EffectInfo] = {}
    res_instrs: dict[BoundInstr, InstrInfo] = {}

    def get_var(v: Var) -> VarInfo:
        assert isinstance(v, Var)
        return res_vars.setdefault(v, VarInfo(v))

    def get_label(v: Label) -> LabelInfo:
        assert isinstance(v, Label)
        return res_labels.setdefault(v, LabelInfo(v))

    def get_effect(loc: EffectBase) -> EffectInfo:
        assert isinstance(loc, EffectBase)
        return res_effects.setdefault(loc, EffectInfo(loc))

    def handle_instr(instr: BoundInstr, parent: BoundInstr | None) -> None:

        children = None
        if unpack.should_unpack(instr):
            children = instr.unpack()
        if children is not None:

            children = list(children)

            for c in children:
                handle_instr(c, parent=instr)

        res_instrs[instr] = InstrInfo(instr, children, parent)

        if children is not None:
            return

        for v in instr.inputs:
            ck_val(v)
            if isinstance(v, Var):

                # for bundles
                # test whether unpack contains a "leaked" variable
                # that is not properly exposed
                if parent:
                    assert v in parent.inputs or v in parent.outputs

                get_var(v).uses.append(instr)

            if isinstance(v, Label):
                if parent:
                    assert v in parent.inputs
                # note: we remove also-defs latter
                get_label(v).uses.append(instr)

        for v in instr.outputs:

            if parent:
                # for bundles
                assert v in parent.outputs

            ck_val(v)
            get_var(v).defs.append(instr)

        # do all remove after, prevent add -> remove -> add
        for l in instr.defines_labels():
            get_label(l).defs.append(instr)
        for l in instr.defines_labels():
            get_label(l).uses.remove(instr)

        for loc in instr.reads():
            get_effect(loc).reads_instrs.append(instr)

        for loc in instr.writes():
            get_effect(loc).writes_instrs.append(instr)

    @f.map_instrs
    def _(instr: BoundInstr):
        handle_instr(instr, None)

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

    return Index(res_vars, res_mvars, res_labels, res_effects, effect_conflicts, res_instrs)


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
            if not x.continues:
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
        if instr.isinst(EmitLabel) or len(instr.jumps_to()) > 0 or not instr.continues:
            return None
        if not instr.is_side_effect_free():
            return None
        for x in instr.outputs:
            if len(index.vars[x].uses) > 0:
                return None
        ans.value = True
        return ()

    return ans.value


_DEAD_VARS: Cell[WeakSet[Var]] = Cell(WeakSet())


@Transform
def rename_private_vars(ctx: TransformCtx) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    _dead = _DEAD_VARS.value
    for v in index.vars:
        assert v not in _dead
        _dead.add(v)

    new_vars = {x.v: mk_var(x.v.type, debug=x.v.debug) for x in index.vars.values()}

    @f.map_instrs
    def _(instr: BoundInstr) -> MapInstrsRes:
        return instr.instr.bind(  # pyright: ignore
            tuple(new_vars[x] for x in instr.outputs),
            *(new_vars[x] if isinstance(x, Var) else x for x in instr.inputs),
        )


_DEAD_LABELS: Cell[WeakSet[Label]] = Cell(WeakSet())


@Transform
def rename_private_labels(ctx: TransformCtx) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    _dead = _DEAD_LABELS.value
    for l in index.labels.values():
        if in_typed(l.v, _dead):
            use_instr = (list(l.uses) + list(l.defs))[0]
            report = use_instr.debug.error(f"use of out-of-scope label {l.v}")
            report.note("this is likely a bug rather than user error")
            report.note("in instruction", use_instr)
            report.note("defined here", l.v.debug)
            report.throw()

    new_labels = {
        x.v: mk_internal_label(x.v.id) if x.private else x.v for x in index.labels.values()
    }

    @f.map_instrs
    def _(instr: BoundInstr) -> MapInstrsRes:
        return instr.instr.bind(  # pyright: ignore
            instr.outputs, *(new_labels[x] if isinstance(x, Label) else x for x in instr.inputs)
        )


_DEAD_MVARS: Cell[WeakSet[MVar]] = Cell(WeakSet())


@Transform
def rename_private_mvars(ctx: TransformCtx) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    _dead = _DEAD_MVARS.value
    for v in index.mvars.values():
        if in_typed(v.v, _dead):
            use_instr = (list(v.uses) + list(v.defs))[0]
            report = use_instr.debug.error(f"use of out-of-scope variable {v.v}")
            report.note("in instruction", use_instr)
            report.note("defined here", v.v.debug)
            report.throw()

    for v in index.mvars.values():
        if v.private:
            _dead.add(v.v)

    new_mvars = {
        x.v: mk_mvar(x.v.type, debug=x.v.debug) if x.private else x.v for x in index.mvars.values()
    }

    def map_fn(instr: BoundInstr) -> MapInstrsRes:
        if i := instr.isinst(Isolate):
            block = Block(list(i.unpack_typed()) + [EndPlaceholder().bind(())], i.debug)
            block.map_instrs(map_fn)
            return Isolate.from_block(block)

        if any(isinstance(x, EffectMvar) for x in instr.reads()):
            if i := instr.isinst(ReadMVar):
                return ReadMVar(new_mvars[i.instr.s]).bind(i.outputs_, *i.inputs_)
            raise TypeError(instr)

        if any(isinstance(x, EffectMvar) for x in instr.writes()):
            if i := instr.isinst(WriteMVar):
                return WriteMVar(new_mvars[i.instr.s]).bind(i.outputs_, *i.inputs_)
            raise TypeError(instr)

    f.map_instrs(map_fn)


@Transform
def mark_all_private_except(ctx: TransformCtx, labels: list[Label]) -> None:
    f = ctx.frag
    index = get_index.call_cached(ctx)

    for l in index.mvars:
        f.scope.private_mvars.add(l)
    for l in index.labels:
        if l not in labels:
            f.scope.private_labels.add(l)
