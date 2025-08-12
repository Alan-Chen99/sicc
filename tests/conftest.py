import pytest

from sicc.config import console_setup
from sicc.config import verbose


@pytest.fixture(autouse=True)
def setup_tests():
    console_setup()
    verbose.value = 0
