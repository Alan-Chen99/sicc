from pathlib import Path

from sicc import Program
from sicc._utils import load_module_from_file
from test_utils import run_one_test

examples_dir = Path(__file__).parent.parent / "examples"


def test_example_explained():
    prog = load_module_from_file(examples_dir / "explained.py", "examples.explained").main
    assert isinstance(prog, Program)
    run_one_test(prog)
