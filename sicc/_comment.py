from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any
from typing import Self
from typing import cast
from typing import override

from rich.pretty import pretty_repr
from rich.text import Text

from ._api import TreeSpec
from ._api import read_uservalue
from ._core import AsRawCtx
from ._core import BoundInstr
from ._core import EffectBase
from ._core import InstrBase
from ._core import RawText
from ._core import TypeList
from ._core import Value
from ._core import format_raw_val
from ._core import format_val
from ._tree_utils import dataclass as optree_dataclass
from ._tree_utils import field as optree_field  # pyright: ignore[reportUnknownVariableType]
from ._utils import ReprAs


@dataclass(frozen=True)
class EffectComment(EffectBase):
    pass


@optree_dataclass
class _CommentStatic:
    text: Text = optree_field(pytree_node=False)


class Comment(InstrBase):
    def __init__(self, tree: TreeSpec[tuple[Any, ...]]) -> None:
        self.tree = tree
        self.in_types = cast(TypeList[tuple[Value, ...]], TypeList(tree.types))
        self.out_types = ()

    def format_with_vals(self, vals: list[Text], prefix: Text) -> Text:
        def random_seq(n: int):
            return "".join(random.choices(string.ascii_letters + string.digits, k=n))

        arg_placeholders = [random_seq(10) for _ in self.in_types]
        args = self.tree.unflatten_unchecked(ReprAs(x) for x in arg_placeholders)

        def handle_placeholders(s: str) -> Text:
            for x, text in zip(arg_placeholders, vals):
                before, sep, after = s.partition(x)
                if sep:
                    return Text() + handle_placeholders(before) + text + handle_placeholders(after)
            return Text(s, "ic10.comment")

        ans = Text()
        ans += prefix
        for arg in args:
            ans += " "
            if isinstance(arg, _CommentStatic):
                ans += arg.text
            else:
                ans += handle_placeholders(pretty_repr(arg, max_width=10000))

        return ans

    @override
    def format(self, instr: BoundInstr[Self]) -> Text:
        vals_text = [format_val(x, typ) for x, typ in zip(instr.inputs_, self.in_types)]
        return self.format_with_vals(vals_text, Text("*", "ic10.jump"))

    def format_raw(self, instr: BoundInstr[Self], ctx: AsRawCtx) -> RawText:
        ans = Text()
        args = [
            format_raw_val(x, ctx, t, instr.debug).text
            for t, x in zip(self.in_types, instr.inputs_)
        ]
        ans += self.format_with_vals(args, Text("#", "ic10.comment"))
        ans += "\n"
        return RawText(ans)

    # dont reorder comments
    reads_ = EffectComment()
    writes_ = EffectComment()


def comment(*args_: Any) -> None:
    args = tuple(
        _CommentStatic(Text(x, "ic10.comment")) if isinstance(x, str) else x for x in args_
    )
    vars, tree = TreeSpec.flatten(args)
    vars_ = [read_uservalue(v) for v in vars]
    return Comment(tree).call(*vars_)
