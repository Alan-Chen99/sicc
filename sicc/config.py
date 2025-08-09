import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Callable
from typing import Iterator

import cappa
import rich.status
import rich.traceback
from cappa import Arg
from cappa import ArgAction
from rich.logging import RichHandler

from ._theme import theme
from ._utils import Cell

if TYPE_CHECKING:
    from rich.console import Console as reconfigure
else:
    from rich import reconfigure

verbose: Cell[int] = Cell(3)

show_src_info: Cell[bool] = Cell(True)

# currently only used in console_setup
# atm binding this does NOT work
console_width: Cell[int | None] = Cell(None)

################################################################################

status_hook: Cell[Callable[[str], None] | None] = Cell(None)

_status_stack: Cell[list[str]] = Cell([])


def _update_status() -> None:
    text = " > ".join(_status_stack.value)
    logging.debug(f"status: {text}")
    if hook := status_hook.get():
        hook(text)


@contextmanager
def with_status(text: str) -> Iterator[None]:
    with _status_stack.bind(_status_stack.value + [text]):
        _update_status()
        yield


@contextmanager
def with_rich_spinner() -> Iterator[None]:
    with rich.status.Status("") as status, status_hook.bind(status.update):
        yield


################################################################################


def console_setup() -> None:
    """
    setup logging, traceback, theme
    """
    format = "%(message)s"
    if verbose.value >= 2:
        level = logging.DEBUG
    elif verbose.value >= 1:
        level = logging.INFO
    else:
        level = logging.WARN

    logging.basicConfig(
        level=level,
        format=format,
        datefmt="[%X]",
        handlers=[RichHandler(show_time=False)],
    )
    reconfigure(theme=theme, width=console_width.value)
    rich.traceback.install()


################################################################################

cappa_group = cappa.Group(name="Config", section=2)


@dataclass
class Config:
    verbose: Annotated[
        int,
        Arg(
            short="-v",
            count=True,
            group=cappa_group,
            propagate=True,
            show_default=False,
            help="does not currrently work correctly; see https://github.com/DanCardin/cappa/issues/232",
        ),
        Arg(long="--verbosity", group=cappa_group, propagate=True),
    ] = 0

    quiet: Annotated[
        int,
        Arg(short="-q", action=ArgAction.count, group=cappa_group, propagate=True),
    ] = 0

    width: Annotated[
        int | None,
        Arg(long=True, group=cappa_group, propagate=True),
    ] = None
    """ terminal width for printing """

    src_info: Annotated[
        bool,
        Arg(
            long="--no-src-info",
            action=ArgAction.store_false,
            show_default=False,
            group=cappa_group,
            propagate=True,
        ),
    ] = True

    def set_vars(self) -> None:
        verbose.value = self.verbose - self.quiet
        console_width.value = self.width
        show_src_info.value = self.src_info
