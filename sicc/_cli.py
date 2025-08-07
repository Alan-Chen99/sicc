from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Annotated
from typing import Callable
from typing import Never

import cappa
import rich
import rich.traceback
from rich import print as print  # autoflake: skip
from rich import reconfigure
from rich.logging import RichHandler

from ._diagnostic import show_pending_diagnostics
from ._theme import theme
from ._tracing import TracedProgram
from ._tracing import trace_program
from .config import verbose


class SuppressExit(Exception):
    def __init__(self, code: int):
        self.code = code


def console_setup():
    """
    setup logging, traceback, theme
    """
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        # level=logging.DEBUG,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(show_time=False)],
    )
    reconfigure(theme=theme)
    rich.traceback.install()


@dataclass
class Program:
    fn: Callable[[], None]

    def trace(self) -> TracedProgram:
        try:
            with trace_program() as res:
                self.fn()
            prog = res.value
            prog.check()
        finally:
            show_pending_diagnostics()
        return prog

    def cli(self) -> Never:
        console_setup()
        cli = cappa.parse(Cli)
        try:
            cli.call(self)
        except SuppressExit as e:
            exit(e.code)

        exit(0)


def program():
    def inner(fn: Callable[[], None]) -> Program:
        return Program(fn)

    return inner


group = cappa.Group(name="Global", section=1)


@cappa.command(name=sys.argv[0])
@dataclass
class Cli:
    cmd: cappa.Subcommands[Asm | Ir | Optimize | None] = None

    verbose: Annotated[
        int,
        cappa.Arg(short="-v", action=cappa.ArgAction.count, group=group, propagate=True),
        cappa.Arg(long="--verbosity", group=group, propagate=True),
    ] = 1

    def call(self, prog: Program) -> None:
        verbose.value = self.verbose
        if self.cmd is None:
            self.cmd = Asm()
        return self.cmd.call(prog, self)


@dataclass
class Ir:
    """
    print internal representation after a subset of optimizations
    """

    def call(self, prog: Program, cli: Cli) -> None:
        ans = prog.trace()
        print("Success; Internal Representation:")
        print(ans)


@dataclass
class Optimize:
    """
    print optimized version. instructions here have one-to-one correspondence to output asm
    """

    def call(self, prog: Program, cli: Cli) -> None:
        f = prog.trace()
        f.optimize()

        print("Success; Optimized:")
        print(f)


@dataclass
class Asm:
    """(default) run the compiler and get assembly"""

    def call(self, prog: Program, cli: Cli) -> None:
        f = prog.trace()
        f.optimize()
        f.regalloc()

        print("Success; Compiled:")
        print(f)
