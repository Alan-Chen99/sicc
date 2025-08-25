from typing import Iterator

from .._api import Comment
from .._core import Block
from .._core import BoundInstr
from .._core import Fragment
from .._diagnostic import add_debug_info
from .._diagnostic import catch_ex_and_exit
from .._instructions import EmitLabel
from .._instructions import EndPlaceholder
from .._instructions import RawInstr
from .._tracing import internal_transform

_LOWER_KEEP = (
    Comment,
    EmitLabel,
    EndPlaceholder,
    RawInstr,
)


def lower_one_instr(instr: BoundInstr) -> list[BoundInstr]:
    if type(instr.instr) in _LOWER_KEEP:
        return [instr]
    with add_debug_info(instr.debug):
        parts = list(instr.instr.lower(instr))
    ans: list[BoundInstr] = []
    for x in parts:
        ans += lower_one_instr(x)
    return ans


def lower_instrs(f: Fragment) -> None:
    """
    calss the .lower method on each instruction; see InstrBase.lower
    """

    with internal_transform(f), catch_ex_and_exit(f):

        def lower_block(b: Block) -> Iterator[BoundInstr]:
            for instr in b.contents:
                yield from lower_one_instr(instr)

        f.blocks = {a: Block(list(lower_block(b)), b.debug) for a, b in f.blocks.items()}
