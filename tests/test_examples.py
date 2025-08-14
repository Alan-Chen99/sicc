from pathlib import Path

from sicc import Program
from sicc._utils import load_module_from_file
from test_utils import run_one_test

examples_dir = Path(__file__).parent.parent / "examples"


def _test_example(name: str):
    mod = load_module_from_file(examples_dir / f"{name}.py", f"examples.{name}")
    programs = {k: p for k, p in mod.__dict__.items() if isinstance(p, Program)}

    (prog,) = programs.values()
    run_one_test(prog)


def test_example_explained():
    _test_example("explained")


def test_example_ic11_compare():
    _test_example("ic11_compare")
