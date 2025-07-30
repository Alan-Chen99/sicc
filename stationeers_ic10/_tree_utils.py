from __future__ import annotations

from typing import TYPE_CHECKING

from optree.pytree import reexport

if TYPE_CHECKING:
    from optree import dataclasses as dataclasses
    from optree import pytree as pytree
else:
    pytree = reexport(namespace=__name__)
    dataclasses = pytree.dataclasses
