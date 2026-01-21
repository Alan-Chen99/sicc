from __future__ import annotations

from dataclasses import dataclass
from typing import Any  # autoflake: skip
from typing import Callable

import rich.repr
from rich import print
from rich.pretty import pretty_repr
from rich.text import Text

from ._api import Variable
from ._api import read_uservalue
from ._core import MVar
from ._core import Scope
from ._diagnostic import register_exclusion
from ._diagnostic import track_caller
from ._tracing import SCOPE_STACK
from ._tracing import mk_mvar
from ._tree_utils import TreeSpec
from ._tree_utils import ValSpec
from ._utils import Empty
from ._utils import empty
from .config import verbose

register_exclusion(__file__)


@dataclass
class State[T = Any]:
    _scope: Scope | None
    _tree: TreeSpec[T] | None = None
    _mvars: list[MVar] | None = None

    def __init__(
        self,
        init: T | Empty = empty,
        *,
        _tree: TreeSpec[T] | None = None,
        _mvars: list[MVar] | None = None,
        _global: bool = False,
        # _scope: Scope | None = None,
    ):
        assert (_tree is None) == (_mvars is None)

        if _global:
            self._scope = None
        else:
            # self._scope = _scope or top_scope_or_err()
            if len(SCOPE_STACK.value) >= 1:
                self._scope = SCOPE_STACK.value[-1]
            else:
                # global var
                self._scope = None

        if _tree:
            self._tree = _tree
            self._mvars = _mvars

        if init is not empty:
            self.write(init)

    def _did_init(self) -> bool:
        assert (self._tree is None) == (self._mvars is None)
        return not (self._tree is None)

    def _log(self, msg: str):
        from ._diagnostic import get_location_desc

        if verbose.value >= 1:
            print(
                Text.assemble(
                    repr(self),
                    " ",
                    (msg, "ic10.str"),
                    "  ",
                    (f"# {get_location_desc()}", "ic10.comment"),
                )
            )

    def __rich_repr__(self) -> rich.repr.Result:
        if self._tree is None:
            yield empty
        else:
            yield self._tree

    def __repr__(self):
        return pretty_repr(self)

    def read(self, *, _log: str = "read") -> T:
        self._log(_log)
        assert self._tree is not None
        assert self._mvars is not None

        with track_caller():
            vars_ = [x.read() for x in self._mvars]
        return self._tree._unflatten_vals_ro(vars_)

    def write(self, v: T, *, _log: str = "write"):
        self._log(_log)

        if self._tree is None:
            _leaves, tree = TreeSpec.flatten(v)
            assert self._mvars is None
            # lazy creation -> all leaves mutable
            self._tree = TreeSpec(
                tree.tree, [ValSpec(type=l.type, mutable=True) for l in tree.leaves]
            )
            self._mvars = [mk_mvar(t) for t in tree.types]

        assert self._mvars is not None

        if any(not x.mutable for x in self._tree.leaves):
            raise TypeError(f"cannot write to {self._tree} because it is not mutable")

        leaves = self._tree.flatten_up_to(v, check_mut=False)
        vars = [read_uservalue(x) for x in leaves]

        with track_caller():
            for mv, arg in zip(self._mvars, vars):
                mv.write(arg)

    @property
    def value(self) -> T:
        return self.read()

    @value.setter
    def value(self, val: T) -> None:
        self.write(val)

    def ref_mut(self) -> T:
        self._log("ref_mut")

        assert self._tree is not None
        assert self._mvars is not None

        return self._tree._unflatten_mvars(self._mvars)

    @staticmethod
    def from_ref_mut[T1](val: T1, /) -> State[T1]:
        leaves, tree = TreeSpec.flatten(val)

        mvars: list[MVar] = []
        for l in leaves:
            if not isinstance(l, Variable):
                raise TypeError(val)
            mvars.append(l._inner)

        return State(_tree=tree, _mvars=mvars)

    def project[R](self, field: Callable[[T], R], /) -> State[R]:
        assert self._tree is not None
        assert self._mvars is not None

        offsets, tree = self._tree.project(field)

        return State(_tree=tree, _mvars=[self._mvars[i] for i in offsets])
