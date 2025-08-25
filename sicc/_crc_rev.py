# pyright: basic, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportWildcardImportFromLibrary=false

import zlib

from z3 import *

POLY = 0xEDB88320  # normal CRC-32 (IEEE 802.3)


# ----------------------------------------------------------------------
# Bit-precise CRC-32 expressed with Z3 bit-vectors
# ----------------------------------------------------------------------
def crc32_update(crc, byte):
    """one byte -> new 32-bit crc (all BitVec expressions)"""
    crc = crc ^ ZeroExt(24, byte)
    for _ in range(8):
        crc = If(crc & 1 == 1, LShR(crc, 1) ^ POLY, LShR(crc, 1))
    return crc


def crc32_expr(sym_bytes, start=0):
    """
    Build a Z3 expression for the CRC-32 of ‘sym_bytes’ starting from ‘start’
    (same convention as zlib.crc32).
    """
    crc = BitVecVal(start ^ 0xFFFFFFFF, 32)
    for b in sym_bytes:
        crc = crc32_update(crc, b)
    return crc ^ 0xFFFFFFFF


# ----------------------------------------------------------------------


def ascii_var(name):
    """8-bit symbolic byte"""
    return BitVec(name, 8)


def char_constraint(b):
    """b is 0-9 or a-z"""
    return Or(
        And(b >= ord("0"), b <= ord("9")),
        And(b >= ord("a"), b <= ord("z")),
        And(b >= ord("A"), b <= ord("Z")),
    )


def crc_rev(prefix: str, wanted_crc: int, length: int = 6) -> str | None:
    """
    find a string with a particular prefix and a crc,
    by appending a suffix of len `length`
    """
    crc_after_prefix = zlib.crc32(prefix.encode()) & 0xFFFFFFFF

    s = Solver()
    # symbolic characters of the suffix
    sym = [ascii_var(f"c{i}") for i in range(length)]
    # constrain alphabet
    for b in sym:
        s.add(char_constraint(b))

    # full crc expression
    crc_expr = crc32_expr(sym, start=crc_after_prefix)
    s.add(crc_expr == wanted_crc)

    if s.check() == sat:
        m = s.model()
        suffix = "".join(chr(m[b].as_long()) for b in sym)
        ans = prefix + suffix
        assert zlib.crc32(ans.encode()) == wanted_crc
        return ans

    raise RuntimeError()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    for i in range(100):
        prefix = f"hello{i}_"
        target = 0x5AE7A6F2
        result = crc_rev(prefix, target)

        print(result or "no match within given length")
