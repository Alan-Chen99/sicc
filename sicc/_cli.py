from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Never

import cappa
from cappa.arg import Arg
from cappa.destructure import Destructured
from rich import print as print  # autoflake: skip
from rich.rule import Rule
from rich.text import Text

from ._api import loop
from ._diagnostic import SuppressExit
from ._diagnostic import describe_fn
from ._diagnostic import register_exclusion
from ._diagnostic import show_pending_diagnostics
from ._tracing import TracedProgram
from ._tracing import trace_program
from ._utils import load_module_from_file
from .config import Config
from .config import console_setup
from .config import with_rich_spinner
from .config import with_status

register_exclusion(__file__)


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
        cli = cappa.parse(CliTied)
        return cli.call(self)


def program(*, loop: bool = False) -> Callable[[Callable[[], None]], Program]:
    def inner(fn: Callable[[], None]) -> Program:
        return Program(fn, loop=loop)

    return inner


@cappa.command(name=sys.argv[0])
@dataclass
class CliTied:
    config: Annotated[Config, Arg(destructured=Destructured(), hidden=True)] = field(
        default_factory=Config
    )
    cmd: cappa.Subcommands[Asm | Ir | Optimize | None] = None

    def call(self, prog: Program) -> Never:
        if self.cmd is None:
            self.cmd = Asm()

        self.config.set_vars()
        console_setup()

        try:
            self.cmd.call(prog)
        except SuppressExit as e:
            exit(e.code)

        exit(0)


@cappa.command(name="sicc")
@dataclass
class Cli:
    file: Path
    """the python file"""

    name: Annotated[str | None, Arg(short="-p", long="--program")] = None
    """program name, if there are multiple"""

    config: Annotated[Config, Arg(destructured=Destructured(), hidden=True)] = field(
        default_factory=Config
    )
    cmd: cappa.Subcommands[Asm | Ir | Optimize | None] = None

    def call(self) -> Never:
        if self.cmd is None:
            self.cmd = Asm()

        self.config.set_vars()
        console_setup()

        mod = load_module_from_file(self.file, self.file.name)

        def _get_program(x: Any) -> Program | None:
            if isinstance(x, Program):
                return x
            if x_ := getattr(x, "_program", None):
                return _get_program(x_)
            return None

        programs = {k: p for k, v in mod.__dict__.items() if (p := _get_program(v))}

        if self.name is None and len(programs) == 1:
            (prog,) = programs.values()
        elif self.name in programs:
            prog = programs[self.name]
        else:
            print("available programs:", list(programs.keys()))
            exit(2)

        try:
            self.cmd.call(prog)
        except SuppressExit as e:
            exit(e.code)

        exit(0)


@dataclass
class Ir:
    """
    print internal representation after a subset of optimizations
    """

    def call(self, prog: Program) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            ans = prog.trace()

        print(Rule(title=Text("Success; Internal Representation:", "ic10.title")))
        print(ans)


@dataclass
class Optimize:
    """
    print optimized version. instructions here have one-to-one correspondence to output asm
    """

    def call(self, prog: Program) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            f = prog.trace()
            f.optimize()

        print(Rule(title=Text("Success; Optimized:", "ic10.title")))
        print(f)


@dataclass
class Asm:
    """(default) run the compiler and get assembly"""

    def call(self, prog: Program) -> None:
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


def main():
    cli = cappa.parse(Cli)
    return cli.call()
