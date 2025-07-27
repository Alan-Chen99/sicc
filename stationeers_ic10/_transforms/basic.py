from dataclasses import dataclass
from dataclasses import field

from .._core import Block
from .._core import BoundInstr
from .._core import Fragment
from .._core import Label
from .._core import MapInstrsRes
from .._core import MVar
from .._core import ReadMVar
from .._core import Var
from .._core import WriteMVar
from .._instructions import EmitLabel
from .._instructions import Jump
from .._tracing import ck_val
from .._tracing import internal_transform
from .._tracing import mk_internal_label
from .._utils import Cell
from .utils import internal_looping_transform


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


@internal_transform
def get_basic_index(f: Fragment) -> BasicIndex:
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
                # set as external if first found
                _ = get_label(v)

        for v in instr.ouputs:
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

    # sanity checks

    for x in res_vars.values():
        assert len(x.defs) == 1

    # private labels actually emitted
    for l in res_labels.values():
        assert len(l.defs) <= 1

    return BasicIndex(res_vars, res_mvars, res_labels)


@internal_transform
def split_blocks(f: Fragment) -> None:
    """
    split blocks based on "continues"

    ensures that there is no labels in the middle of any block
    """

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


@internal_looping_transform
def remove_unused_side_effect_free(f: Fragment) -> bool:
    index = get_basic_index(f)

    ans = Cell(False)

    @f.map_instrs
    def _(instr: BoundInstr) -> MapInstrsRes:
        if instr.isinst(EmitLabel) or instr.does_jump() or not instr.instr.continues:
            return None
        if not instr.is_side_effect_free():
            return None
        for x in instr.ouputs:
            if len(index.vars[x].uses) > 0:
                return None
        ans.value = True
        return []

    return ans.value
