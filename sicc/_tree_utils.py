from __future__ import annotations

from typing import TYPE_CHECKING

from optree.pytree import reexport

if TYPE_CHECKING:
    from dataclasses import dataclass as dataclass

    from optree import dataclasses as dataclasses
    from optree import pytree as pytree

    field = dataclasses.field  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
else:
    pytree = reexport(namespace=__name__)
    dataclasses = pytree.dataclasses
    dataclass = dataclasses.dataclass
    field = dataclasses.field
