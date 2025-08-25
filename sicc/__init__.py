# autoflake: skip_file

# modules
from . import devices as devices
from . import functions as functions

# high level
from ._cli import Program as Program
from ._cli import program as program
from ._diagnostic import show_pending_diagnostics as show_pending_diagnostics

# types
from ._api import EnumEx as Enum  # pyright: ignore[reportUnusedImport]
from ._api import UserValue as UserValue
from ._api import Variable as Variable
from ._api import VarRead as VarRead
from ._core import Label as Label
from ._core import nan as nan
from ._stationeers import BatchMode as BatchMode
from ._stationeers import Color as Color
from ._stationeers import LogicType as LogicType

# units
MPa = 1000

# UserValue aliases
from ._api import Bool as Bool
from ._api import Float as Float
from ._api import Int as Int
from ._api import Str as Str
from ._api import ValLabelLike as ValLabelLike
from ._stationeers import ValBatchMode as ValBatchMode
from ._stationeers import ValLogicTypeLike as ValLogicTypeLike

# control flow primitives
from ._api import asm as asm
from ._api import asm_block as asm_block
from ._api import asm_fn as asm_fn
from ._api import branch as branch
from ._api import cond as cond
from ._api import jump as jump
from ._api import label_ref as label_ref
from ._tracing import exit_program as exit_program
from ._tracing import label as label

# control flow high level
from ._api import block as block
from ._api import cjump as cjump
from ._api import if_ as if_
from ._api import inline_subr as inline_subr
from ._api import loop as loop
from ._api import range_ as range_
from ._api import while_ as while_
from ._tracing import break_ as break_
from ._tracing import continue_ as continue_
from ._tracing import else_ as else_

# subroutines
from ._api import return_ as return_
from ._api import subr as subr

# dataclasses
from ._api import State as State
from ._tree_utils import dataclass as dataclass
from ._tree_utils import dataclasses as dataclasses
from ._tree_utils import pytree as pytree

field = dataclasses.field  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

# devices
from ._stationeers import Device as Device
from ._stationeers import DeviceBase as DeviceBase
from ._stationeers import DeviceTyped as DeviceTyped
from ._stationeers import Pin as Pin

db = Pin.db()
d0 = Pin(0)
d1 = Pin(1)
d2 = Pin(2)
d3 = Pin(3)
d4 = Pin(4)
d5 = Pin(5)

# stack
from ._api import Pointer as Pointer
from ._api import stack_var as stack_var

# functions
from ._api import comment as comment
from ._api import select as select
from ._api import undef as undef
from ._stationeers import yield_ as yield_

# crc32
from ._crc import crc_append as crc_append
from ._crc_rev import crc_rev as crc_rev
from ._utils import crc32 as crc32

# diagnostics
from ._diagnostic import Warnings as Warnings
from ._diagnostic import suppress_warnings as suppress_warnings
