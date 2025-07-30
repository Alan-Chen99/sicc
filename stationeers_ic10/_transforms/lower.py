from typing import Iterator
from typing import Sequence

from .._core import Block
from .._core import BoundInstr
from .._core import InstrBase
from .._core import ReadMVar
from .._core import WriteMVar
from .._instructions import EmitLabel
from .._instructions import RawInstr
from .._tracing import internal_trace_as_rep
from .._utils import Cell
from .utils import LoopingTransform
from .utils import TransformCtx

_LOWER_KEEP_DEFAULT = (
    EmitLabel,
    RawInstr,
    ReadMVar,
    WriteMVar,
)


@LoopingTransform
def lower_instrs(
    ctx: TransformCtx, keep_types: Sequence[type[InstrBase]] = _LOWER_KEEP_DEFAULT
) -> bool:
    # TODO: consider running all "lower" functions and subs vars together in some way

    f = ctx.frag

    keep_types = list(keep_types)
    changed = Cell(False)

    def lower_block(b: Block) -> Iterator[BoundInstr]:
        for instr in b.contents:
            if any(isinstance(instr.instr, typ) for typ in keep_types):
                yield instr
            else:
                changed.value = True
                yield from internal_trace_as_rep(instr, instr.instr.lower)

    f.blocks = {a: Block(list(lower_block(b)), b.debug) for a, b in f.blocks.items()}

    return changed.value
