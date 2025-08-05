from ._tree_utils import dataclasses as dataclasses
from ._tree_utils import pytree as pytree

dataclass = dataclasses.dataclass
field = dataclasses.field  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
