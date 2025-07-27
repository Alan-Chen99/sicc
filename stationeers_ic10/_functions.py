from __future__ import annotations

from ._core import InteralBool
from ._core import InternalValLabel
from ._instructions import Branch
from ._instructions import Jump
from ._instructions import PredVar
from ._instructions import UnreachableChecked

jump = Jump().call
unreachable_checked = UnreachableChecked().call
# lt = PredLT().call
# le = PredLE().call


def branch(cond: InteralBool, on_true: InternalValLabel, on_false: InternalValLabel) -> None:
    return Branch(PredVar()).call(on_true, on_false, cond)
