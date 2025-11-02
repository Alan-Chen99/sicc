from pathlib import Path
from typing import Callable

from rich import print as print
from rich.rule import Rule
from rich.text import Text

from sicc import Program
from sicc import show_pending_diagnostics
from sicc._diagnostic import describe_fn
from sicc.config import with_rich_spinner
from sicc.config import with_status

OUTPUT_DIR = Path(__file__).parent / "outputs"


def run_one_test(prog: Program):
    with with_rich_spinner(), with_status(f"{describe_fn(prog.fn)}"):
        with with_status("Trace"):
            f = prog.trace()
        with with_status("Optimize"):
            f.optimize()
        with with_status("Optimize (final)"):
            f.optimize_final()
        with with_status("Regalloc"):
            f.regalloc()

    # TODO: put diagnostics in output
    show_pending_diagnostics()

    output = f.gen_asm().text

    print(Rule(title=Text("Success; Output (readable):", "ic10.title")))
    print(f)
    print(Rule(title=Text("Raw Equivalent:", "ic10.title")))
    print(output)
    print(Rule())

    (OUTPUT_DIR / f"{describe_fn(prog.fn)}.out").write_text(output.plain + "\n")


def wrap_test(prog: Program) -> Callable[[], None]:
    def inner():
        run_one_test(prog)

    inner._program = prog  # pyright: ignore[reportFunctionMemberAccess]
    return inner
