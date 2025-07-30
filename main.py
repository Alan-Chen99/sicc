import gc
from pathlib import Path

from rich import print
from rich import reconfigure
from rich.panel import Panel
from rich.pretty import Pretty
from rich.pretty import pretty_repr
from rich.theme import Theme
from rich.themes import DEFAULT
from rich.traceback import install

from stationeers_ic10 import dataclass
from stationeers_ic10 import dataclasses
from stationeers_ic10 import pytree
from stationeers_ic10._api import Float
from stationeers_ic10._api import Int
from stationeers_ic10._api import State
from stationeers_ic10._api import Variable
from stationeers_ic10._api import if_
from stationeers_ic10._api import return_
from stationeers_ic10._api import subr
from stationeers_ic10._api import trace_to_subr
from stationeers_ic10._core import MVar
from stationeers_ic10._diagnostic import check_must_use
from stationeers_ic10._instructions import AddF
from stationeers_ic10._instructions import PredLT
from stationeers_ic10._stationeers import BatchMode
from stationeers_ic10._theme import theme
from stationeers_ic10._tracing import ensure_label
from stationeers_ic10._tracing import label
from stationeers_ic10._tracing import trace_if
from stationeers_ic10._tracing import trace_main_test
from stationeers_ic10._transforms import emit_asm
from stationeers_ic10.functions import black_box
from stationeers_ic10.functions import jump


@dataclass
class Data:
    x: Float
    y: Float


@subr()
def helper2(x: Int) -> Int:
    # with if_(x > 0):
    #     return_(Data(7, x))
    # return_(Data(x, 9))

    # return x + 111
    # with if_(x > 0):
    #     return_(x)

    return x + 5


@subr()
def helper(x: Int) -> Int:
    # with if_(x > 0):
    #     return_(Data(7, x))
    # return_(Data(x, 9))

    x = helper2(x)

    # with if_(x > 0):
    #     return_(x + 5)

    return x + 7


def main():
    reconfigure(theme=theme)

    with trace_main_test() as res:
        label("start")

        x = black_box(1)
        x = Variable(helper2(x))
        with if_(x > 1):
            x.value = helper2(x)

        black_box(x)

        # x = helper(1)

        # black_box(helper(1))
        # black_box(helper(1))
        # black_box(helper(1))
        # black_box(helper(1))
        # # with if_(black_box(True) > 0):
        # #     s.write(Data(black_box(2), 1))
        # #     # s.value = Data(3, 1)

        # with if_(s.read().x > 5):
        #     s.value = Data(s.value.y, s.value.x)

        # black_box(s.read().x)
        # black_box(s.read().y)

        jump("start")

    return res.value


if __name__ == "__main__":

    # install(show_locals=True)
    install()
    ans = main()
    # for b in ans.blocks.values():
    #     for x in b.contents:
    #         print(x)

    print()
    print()
    print()

    gc.collect()
    check_must_use()

    # emit_asm(ans)

    print(ans)
