from dataclasses import dataclass
from dataclasses import field

from ordered_set import OrderedSet

from .._core import Block
from .._core import BoundInstr
from .._core import Label
from .._core import MapInstrsRes
from .._core import MVar
from .._core import ReadMVar
from .._core import Value
from .._core import Var
from .._core import WriteMVar
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
class BasicIndex:
    vars: dict[Var, VarInfo]
    mvars: dict[MVar, MVarInfo]

    labels: dict[Label, LabelInfo]


@CachedFn
def get_basic_index(ctx: TransformCtx) -> BasicIndex:
    f = ctx.frag
    res_vars: dict[Var, VarInfo] = {}
    res_mvars: dict[MVar, MVarInfo] = {}
    res_labels: dict[Label, LabelInfo] = {}

    def get_var(v: Var) -> VarInfo:
        assert isinstance(v, Var)
        return res_vars.setdefault(v, VarInfo(v, [], []))

    def get_mvar(v: MVar) -> MVarInfo:
        assert isinstance(v, MVar)
        return res_mvars.setdefault(v, MVarInfo(v))

    def get_label(v: Label) -> LabelInfo:
        assert isinstance(v, Label)
        return res_labels.setdefault(v, LabelInfo(v))

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

        if i := instr.isinst(ReadMVar):
            get_mvar(i.instr.s).uses.append(instr)
        if i := instr.isinst(WriteMVar):
            get_mvar(i.instr.s).defs.append(instr)

        if i := instr.isinst(EmitLabel):
            (arg,) = i.inputs_
            assert not isinstance(arg, Var)
            l = get_label(arg)
            l.defs.append(instr)

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

    return BasicIndex(res_vars, res_mvars, res_labels)


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
    index = get_basic_index.call_cached(ctx)

    ans = Cell(False)

    @f.map_instrs
    def _(instr: BoundInstr):
        if instr.isinst(EmitLabel) or instr.does_jump() or not instr.instr.continues:
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
def rename_private(ctx: TransformCtx) -> None:
    f = ctx.frag
    index = get_basic_index.call_cached(ctx)

    new_labels = {
        x.v: mk_internal_label(x.v.id) if x.private else x.v for x in index.labels.values()
    }
    new_vars = {x.v: mk_var(x.v.type, debug=x.v.debug) for x in index.vars.values()}

    new_mvars = {
        x.v: mk_mvar(x.v.type, debug=x.v.debug) if x.private else x.v for x in index.mvars.values()
    }

    @f.map_instrs
    def _(instr: BoundInstr) -> MapInstrsRes:
        def map_val(x: Value) -> Value:
            if isinstance(x, Label):
                return new_labels[x]
            if isinstance(x, Var):
                return new_vars[x]
            return x

        if i := instr.isinst(ReadMVar):
            (out_var,) = i.outputs_
            return ReadMVar(new_mvars[i.instr.s]).bind((new_vars[out_var],))
        if i := instr.isinst(WriteMVar):
            (in_var,) = i.inputs_
            return WriteMVar(new_mvars[i.instr.s]).bind((), map_val(in_var))
        return instr.instr.bind(  # pyright: ignore
            tuple(new_vars[x] for x in instr.outputs), *(map_val(x) for x in instr.inputs)
        )


@Transform
def mark_all_private_except(ctx: TransformCtx, labels: list[Label]) -> None:
    f = ctx.frag
    index = get_basic_index.call_cached(ctx)

    for l in index.mvars:
        f.scope.private_mvars.add(l)
    for l in index.labels:
        if l not in labels:
            f.scope.private_labels.add(l)
