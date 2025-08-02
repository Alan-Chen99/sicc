import functools
import gc
import logging
from pathlib import Path

from rich import print
from rich import reconfigure
from rich.logging import RichHandler
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
from stationeers_ic10._core import FORMAT_ANNOTATE
from stationeers_ic10._core import AlwaysUnpack
from stationeers_ic10._core import MVar
from stationeers_ic10._core import NeverUnpack
from stationeers_ic10._diagnostic import check_must_use
from stationeers_ic10._diagnostic import show_pending_diagnostics
from stationeers_ic10._instructions import AddF
from stationeers_ic10._instructions import PredLT
from stationeers_ic10._stationeers import BatchMode
from stationeers_ic10._theme import theme
from stationeers_ic10._tracing import ensure_label
from stationeers_ic10._tracing import isolate
from stationeers_ic10._tracing import label
from stationeers_ic10._tracing import trace_if
from stationeers_ic10._tracing import trace_main_test
from stationeers_ic10._transforms import emit_asm
from stationeers_ic10._transforms.control_flow import build_control_flow_graph
from stationeers_ic10._transforms.control_flow import compute_label_provenance
from stationeers_ic10.functions import black_box
from stationeers_ic10.functions import jump


@dataclass
class Data:
    x: Float
    y: Float


@subr()
def child(x: Int) -> Int:
    # with if_(x > 0):
    #     return_(Data(7, x))
    # return_(Data(x, 9))

    # return x + 111
    # with if_(x > 0):
    #     return_(x)

    return x + 5


@subr()
def parent(x: Int) -> Int:
    # with if_(x > 0):
    #     return_(Data(7, x))
    # return_(Data(x, 9))

    # x += 7
    # x = child(x)

    with if_(x > 0):
        return_(x + 5)

    return child(x)


def main():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level=logging.INFO,
        # level=logging.DEBUG,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(show_time=False)],
    )
    reconfigure(theme=theme)

    with trace_main_test() as res:
        label("start")

        # x = Variable(int)

        # with if_(black_box(True)):
        #     x.value += 1

        black_box(child(1))
        black_box(parent(2))
        black_box(parent(3))

        # x = black_box(child(1))
        # with if_(x > 1):
        #     x.value = 2
        # with if_(x > 2):
        #     black_box(child(x))

        # black_box(x)

        # # vx.value = 5
        # x = Variable(5)

        # with if_(x > 2):
        #     with if_(x > 3):
        #         jump("start")
        #     x.value = child(x + 11)
        #     # x += 1

        # black_box(x)

        # with isolate():
        #     x = black_box(1)
        #     with isolate():
        #         x.value = child(x)
        #     x = black_box(x)

        # black_box(x)

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
    try:
        ans = main()
        # for b in ans.blocks.values():
        #     for x in b.contents:
        #         print(x)

        print()
        print()
        print()

        check_must_use()
    finally:
        show_pending_diagnostics()

    # emit_asm(ans)

    res = compute_label_provenance(ans, out_unpack=NeverUnpack(), analysis_unpack=AlwaysUnpack())
    # res = compute_label_provenance(ans, out_unpack=AlwaysUnpack(), analysis_unpack=AlwaysUnpack())
    with FORMAT_ANNOTATE.bind(res.annotate):
        print(ans)

    # graph = build_control_flow_graph(ans, out_unpack=NeverUnpack(), analysis_unpack=AlwaysUnpack())
    # print(list(graph.edges))


@functools.cache
def testfn(x: int):
    pass
