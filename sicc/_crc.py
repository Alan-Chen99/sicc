from ._api import Int
from ._api import Str
from ._api import Variable
from ._api import if_
from ._api import range_
from ._api import subr


@subr
def crc_append(s: Str, b: Int) -> Str:
    crc_ = Variable(s).transmute(int)

    # clear the left 32 bits
    # (they are all 1 if the hash is given as negative)
    crc_ &= 2**32 - 1

    crc_ ^= 0xFFFFFFFF
    crc_ ^= b

    crc = Variable(crc_)

    for _idx in range_(8):
        right_bit = crc & 1
        crc.value >>= 1
        with if_(right_bit != 0):
            crc.value ^= 0xEDB88320

    ans = Variable(crc ^ 0xFFFFFFFF)

    # normalize back as negative
    with if_(ans >= 2**31):
        ans.value -= 2**32

    return ans.transmute(str)
