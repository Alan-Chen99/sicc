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
            programs = [p for p in mod.__dict__.values() if isinstance(p, Program)]
            assert programs, f"{f.name} defines no program"

            for prog in programs:
                run_one_test(prog)
