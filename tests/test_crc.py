from rich import print as print

from sicc import *
from sicc.devices import *
from test_utils import wrap_test


@wrap_test
@program
def test_crc_append():
    for s in ["aa", "bb", "cc"]:
        ans = crc_append(s[:-1], ord(s[-1]))
        yield_()
        comment("res:", ans, "should be:", crc32(s))


def test_crc_rev():
    for i in range(10):
        prefix = f"hello{i}_"
        target = 300 + i
        crc_rev(prefix, target)
