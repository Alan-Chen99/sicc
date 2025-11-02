import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Iterator

import cappa
import rich.status
import rich.traceback
from cappa import Arg
from cappa import ArgAction
from rich import print as rich_print
from rich.console import Console
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
# atm binding these does NOT work
console_width: Cell[int | None] = Cell(None)
log_file: Cell[Path | None] = Cell(None)

################################################################################

status_hook: Cell[Callable[[str], None] | None] = Cell(None)

_status_stack: Cell[list[str]] = Cell([])


def _status_text() -> str:
    return " > ".join(_status_stack.value)


def _update_status() -> None:
    text = _status_text()
    logging.debug(f"status: {text}")
    if hook := status_hook.get():
        hook(text)


@contextmanager
def with_status(text: str) -> Iterator[None]:
    with _status_stack.bind(_status_stack.value + [text]):
        _update_status()
        yield


RICH_SPINNER: Cell[rich.status.Status] = Cell()


@contextmanager
def with_rich_spinner() -> Iterator[None]:
    with (
        rich.status.Status("") as status,
        RICH_SPINNER.bind(status),
        status_hook.bind(status.update),
    ):
        yield


################################################################################


def console_setup() -> None:
    """
    setup logging, traceback, theme
    """
    # ensure "register_exclusion" all gets called
    from ._diagnostic import TRACEBACK_SUPPRESS

    format = "%(message)s"
    if verbose.value >= 3:
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

    if f := log_file.value:
        # clear the file
        with f.open("w"):
            pass

    reconfigure(theme=theme, width=console_width.value)
    if verbose.value >= 1:
        rich.traceback.install()
    else:
        rich.traceback.install(suppress=TRACEBACK_SUPPRESS.value)


def print(*objects: Any, sep: str = " ", end: str = "\n") -> None:
    rich_print(*objects, sep=sep, end=end)

    if f := log_file.value:
        with f.open("a") as stream:
            console = Console(file=stream, theme=theme, width=console_width.value or 100)
            console.print(*objects, sep=sep, end=end)


print_ = print

################################################################################

cappa_group = cappa.Group(name="Config", section=2)


@dataclass
class Config:
    verbose: Annotated[
        int,
        Arg(short="-v", count=True, group=cappa_group, propagate=True, show_default=False),
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

    output: Annotated[
        Path | None,
        Arg(long=True, short=True, group=cappa_group, propagate=True),
    ] = None
    """ write output also to a file """

    raw_output: Annotated[
        Path | None,
        Arg(long=True, short=True, group=cappa_group, propagate=True),
    ] = None
    """ write only the final raw output to file """

    def set_vars(self) -> None:
        verbose.value = self.verbose - self.quiet
        console_width.value = self.width
        show_src_info.value = self.src_info
        log_file.value = self.output
