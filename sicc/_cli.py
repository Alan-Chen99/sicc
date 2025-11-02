from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Callable
from typing import Never
from typing import TypedDict
from typing import Unpack
from typing import overload

import cappa
from cappa import Arg
from cappa import Destructured
from rich.rule import Rule
from rich.text import Text

from ._comment import comment
from ._control_flow import loop
from ._core import TracedProgram
from ._diagnostic import SuppressExit
from ._diagnostic import describe_fn
from ._diagnostic import register_exclusion
from ._diagnostic import show_pending_diagnostics
from ._subr import FunctionRet
from ._subr import inline_subr
from ._tracing import trace_program
from ._utils import load_module_from_file
from .config import Config
from .config import console_setup
from .config import print as print
from .config import with_rich_spinner
from .config import with_status

register_exclusion(__file__)


@dataclass
class Program:
    fn: Callable[[], FunctionRet[None]]
    loop: bool = False
    header: str | None = None

    def trace(self) -> TracedProgram:
        fn_ = inline_subr(self.fn)

        try:
            with trace_program() as res:
                if self.header is not None:
                    comment(self.header)

                if self.loop:
                    with loop():
                        fn_()
                else:
                    fn_()

            prog = res.value
            prog.check()
        finally:
            show_pending_diagnostics()
        return prog

    def cli(self) -> Never:
        cli = cappa.parse(CliTied)
        return cli.call(self)


class ProgramOpts(TypedDict, total=False):
    loop: bool
    header: str | None


@overload
def program(
    **kwargs: Unpack[ProgramOpts],
) -> Callable[[Callable[[], FunctionRet[None]]], Program]: ...
@overload
def program(func: Callable[[], FunctionRet[None]], /, **kwargs: Unpack[ProgramOpts]) -> Program: ...


def program(
    func: Callable[[], FunctionRet[None]] | None = None, /, **kwargs: Unpack[ProgramOpts]
) -> Callable[[Callable[[], FunctionRet[None]]], Program] | Program:
    def inner(fn: Callable[[], FunctionRet[None]]) -> Program:
        return Program(fn, **kwargs)

    if func is None:
        return inner
    return inner(func)


@cappa.command(name=sys.argv[0])
@dataclass
class CliTied:
    config: Annotated[Destructured[Config], Arg(hidden=True)] = field(default_factory=Config)
    cmd: cappa.Subcommands[Asm | Trace | Ir | Optimize | None] = None

    def call(self, prog: Program) -> Never:
        if self.cmd is None:
            self.cmd = Asm()

        self.config.set_vars()
        console_setup()

        try:
            self.cmd.call(prog, self.config)
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

    config: Annotated[Destructured[Config], Arg(hidden=True)] = field(default_factory=Config)
    cmd: cappa.Subcommands[Asm | Trace | Ir | Optimize | None] = None

    def call(self) -> Never:
        if self.cmd is None:
            self.cmd = Asm()

        self.config.set_vars()
        console_setup()

        mod = load_module_from_file(self.file, self.file.stem)

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
            self.cmd.call(prog, self.config)
        except SuppressExit as e:
            exit(e.code)

        exit(0)


@dataclass
class Trace:
    def call(self, prog: Program, config: Config) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            ans = prog.trace()

        print(Rule(title=Text("Success; Internal Representation:", "ic10.title")))
        print(ans)


@dataclass
class Ir:
    """
    print internal representation after a subset of optimizations
    """

    def call(self, prog: Program, config: Config) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            f = prog.trace()
            f.optimize()

        print(Rule(title=Text("Success; Internal Representation:", "ic10.title")))
        print(f)


@dataclass
class Optimize:
    """
    print optimized version. instructions here have one-to-one correspondence to output asm
    """

    def call(self, prog: Program, config: Config) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            f = prog.trace()
            f.optimize()
            f.optimize_final()

        print(Rule(title=Text("Success; Optimized:", "ic10.title")))
        print(f)


@dataclass
class Asm:
    """(default) run the compiler and get assembly"""

    def call(self, prog: Program, config: Config) -> None:
        with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
            with with_status("Trace"):
                f = prog.trace()
            with with_status("Optimize"):
                f.optimize()
            with with_status("Optimize (final)"):
                f.optimize_final()
            with with_status("Regalloc"):
                f.regalloc()

        show_pending_diagnostics()

        print(Rule(title=Text("Success; Output (readable):", "ic10.title")))
        print(f)
        print(Rule(title=Text("Raw Equivalent:", "ic10.title")))
        out_text = f.gen_asm().text
        print(out_text)
        print(Rule())

        if config.raw_output:
            config.raw_output.write_text(out_text.plain)


def main():
    cli = cappa.parse(Cli)
    return cli.call()
