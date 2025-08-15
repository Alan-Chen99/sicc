from pathlib import Path

from pytest_subtests import SubTests

from sicc import Program
from sicc._utils import load_module_from_file
from test_utils import run_one_test

examples_dir = Path(__file__).parent.parent / "examples"


def test_examples(subtests: SubTests):
    for f in examples_dir.iterdir():
        if f.is_dir() or f.suffix != ".py":
            continue

        with subtests.test(msg=f.name):
            mod = load_module_from_file(f, f"examples.{f.stem}")
            programs = {k: p for k, p in mod.__dict__.items() if isinstance(p, Program)}

            (prog,) = programs.values()
            run_one_test(prog)
