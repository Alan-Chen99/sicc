from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import field
from typing import Annotated
from typing import Callable
from typing import Never

import cappa
from cappa.arg import Arg
from cappa.destructure import Destructured
from rich import print as print  # autoflake: skip
from rich.rule import Rule
from rich.text import Text

from ._api import loop
from ._diagnostic import describe_fn
from ._diagnostic import register_exclusion
from ._diagnostic import show_pending_diagnostics
from ._tracing import TracedProgram
from ._tracing import trace_program
from .config import Config
from .config import console_setup
from .config import with_rich_spinner
from .config import with_status

register_exclusion(__file__)


class SuppressExit(Exception):
    def __init__(self, code: int):
        self.code = code


@dataclass
class Program:
    fn: Callable[[], None]
    loop: bool

    def trace(self) -> TracedProgram:
        try:
            with trace_program() as res:
                if self.loop:
                    with loop():
                        self.fn()
                else:
                    self.fn()

            prog = res.value
            prog.check()
        finally:
            show_pending_diagnostics()
        return prog

    def cli(self) -> Never:
        cli = cappa.parse(Cli)
        cli.config.set_vars()
        console_setup()

        try:
            cli.call(self)
        except SuppressExit as e:
            exit(e.code)

        exit(0)


def program(loop: bool = False):
    def inner(fn: Callable[[], None]) -> Program:
        return Program(fn, loop=loop)

    return inner


group = cappa.Group(name="Global", section=1)


@cappa.command(name=sys.argv[0])
@dataclass
class Cli:
    config: Annotated[Config, Arg(destructured=Destructured(), hidden=True)] = field(
        default_factory=Config
    )
    cmd: cappa.Subcommands[Asm | Ir | Optimize | None] = None

    def call(self, prog: Program) -> None:
        if self.cmd is None:
            self.cmd = Asm()
        return self.cmd.call(prog, self)


@dataclass
class Ir:
    """
    print internal representation after a subset of optimizations
    """

    def call(self, prog: Program, cli: Cli) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            ans = prog.trace()

        print(Rule(title=Text("Success; Internal Representation:", "ic10.title")))
        print(ans)


@dataclass
class Optimize:
    """
    print optimized version. instructions here have one-to-one correspondence to output asm
    """

    def call(self, prog: Program, cli: Cli) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            f = prog.trace()
            f.optimize()

        print(Rule(title=Text("Success; Optimized:", "ic10.title")))
        print(f)


@dataclass
class Asm:
    """(default) run the compiler and get assembly"""

    def call(self, prog: Program, cli: Cli) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            with with_status("Trace"):
                f = prog.trace()
            with with_status("Optimize"):
                f.optimize()
            with with_status("Regalloc"):
                f.regalloc()

        show_pending_diagnostics()

        print(Rule(title=Text("Success; Output (readable):", "ic10.title")))
        print(f)
        print(Rule(title=Text("Raw Equivalent:", "ic10.title")))
        print(f.gen_asm().text)
        print(Rule())
