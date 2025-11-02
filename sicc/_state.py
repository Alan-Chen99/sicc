from __future__ import annotations

from dataclasses import dataclass
from typing import Any  # autoflake: skip
from typing import Callable

import rich.repr
from rich.pretty import pretty_repr

from ._api import TreeSpec
from ._api import Variable
from ._api import read_uservalue
from ._core import MVar
from ._core import Scope
from ._diagnostic import track_caller
from ._tracing import _CUR_SCOPE
from ._tracing import mk_mvar
from ._utils import empty
from ._utils import empty_t


@dataclass
class State[T = Any]:
    _scope: Scope | None
    _tree: TreeSpec[T] | None = None
    _vars: list[MVar] | None = None

    def __init__(self, init: T | empty_t = empty, *, _tree: TreeSpec[T] | None = None):
        self._scope = _CUR_SCOPE.get()
        if _tree:
            self._tree = _tree
            self._vars = [mk_mvar(t) for t in _tree.types]
        if not isinstance(init, empty_t):
            self.write(init)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self._tree

    def __repr__(self):
        return pretty_repr(self)

    def read(self) -> T:
        assert self._tree is not None
        assert self._vars is not None

        with track_caller():
            vars_ = [x.read() for x in self._vars]
        return self._tree._unflatten_vals_ro(vars_)

    def write(self, v: T):
        if self._tree is None:
            _leaves, tree = TreeSpec.flatten(v)
            assert self._vars is None
            self._tree = tree
            if self._scope:
                with _CUR_SCOPE.bind(self._scope):
                    self._vars = [mk_mvar(t) for t in tree.types]
            else:
                with _CUR_SCOPE.bind_clear():
                    self._vars = [mk_mvar(t) for t in tree.types]

        leaves = self._tree.flatten_up_to(v)
        vars = [read_uservalue(x) for x in leaves]

        assert self._vars is not None
        with track_caller():
            for mv, arg in zip(self._vars, vars):
                mv.write(arg)

    @property
    def value(self) -> T:
        return self.read()

    @value.setter
    def value(self, val: T) -> None:
        self.write(val)

    def ref_mut(self) -> T:
        assert self._tree is not None
        assert self._vars is not None

        return self._tree.unflatten(Variable(x.type, _mvar=x) for x in self._vars)

    def project[R](self, field: Callable[[T], R], /) -> State[R]:
        assert self._tree is not None
        assert self._vars is not None

        offsets, tree = self._tree.project(field)

        ans = State(_tree=tree)
        ans._vars = [self._vars[i] for i in offsets]
        return ans
