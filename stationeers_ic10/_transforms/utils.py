from typing import Callable
from typing import Concatenate

from .._core import Fragment
from .._tracing import internal_transform


def _run_phases_once(frag: Fragment, *fs: Callable[[Fragment], bool | None]) -> bool:
    changed = False

    for f in fs:
        ans = f(frag)
        if ans is not None:
            assert isinstance(ans, bool)
            changed = changed or ans

    return changed


def run_phases(frag: Fragment, *fs: Callable[[Fragment], bool | None], loop: bool = True) -> bool:
    if not loop:
        return _run_phases_once(frag, *fs)
    return internal_looping_transform(_run_phases_once)(frag, *fs)


def internal_looping_transform[**P](
    fn: Callable[Concatenate[Fragment, P], bool],
) -> Callable[Concatenate[Fragment, P], bool]:

    @internal_transform
    def inner(frag: Fragment, *args: P.args, **kwargs: P.kwargs) -> bool:
        changed = False
        while fn(frag, *args, **kwargs):
            changed = True
        return changed

    return inner
