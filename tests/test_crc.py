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


@wrap_test
@program
def test_names_by_hash():
    # names whose hashes are first, first + 1, ...; negative, as crc32 and IC10 give hashes
    # of 2**31 and up
    first = -1000000000
    names = [crc_rev(f"mem{i}_", first + i) for i in range(3)]
    LogicMemory(names[2]).Setting = 2
    # the str view of the loop index shares its register: no move in the loop
    for h in range_(first, first + len(names)):
        LogicMemory(h.transmute(str)).Setting = h
