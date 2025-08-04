import functools
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
from stationeers_ic10._tracing import label
from stationeers_ic10._tracing import trace_bundle
from stationeers_ic10._tracing import trace_if
from stationeers_ic10._tracing import trace_main_test
from stationeers_ic10._transforms import emit_asm
from stationeers_ic10._transforms.control_flow import build_control_flow_graph
from stationeers_ic10._transforms.control_flow import compute_label_provenance
from stationeers_ic10._transforms.regalloc import compute_lifetimes_all
from stationeers_ic10.functions import black_box
from stationeers_ic10.functions import jump


@dataclass
class Data:
    x: Float
    y: Float


@subr()
def child(x: Int):
    with if_(x > 0):
        return_(Data(7, x))
    return Data(x, 9)


@subr()
def parent(x: Int) -> Int:
    with if_(x > 0):
        return_(child(x).x)

    return x + 5


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

        x = black_box(parent(1))
        with if_(x > 1):
            parent(x + 51)

        parent(1)

        # with trace_bundle():
        with if_(x > 2):
            black_box(parent(x))

        black_box(55)

        jump("start")

    return res.value


if __name__ == "__main__":

    # install(show_locals=True)
    install()
    try:
        ans = main()
        print()
        print()
        print()

        check_must_use()
    finally:
        show_pending_diagnostics()

    # emit_asm(ans)

    res = compute_lifetimes_all(ans)
    with res.with_anno():
        print(ans)

    # res = compute_label_provenance(ans, out_unpack=NeverUnpack(), analysis_unpack=AlwaysUnpack())
    # res = compute_label_provenance(ans, out_unpack=AlwaysUnpack(), analysis_unpack=AlwaysUnpack())
    # with FORMAT_ANNOTATE.bind(res.annotate):
    #     print(ans)
    # print(ans)

    # graph = build_control_flow_graph(ans, out_unpack=NeverUnpack(), analysis_unpack=AlwaysUnpack())
    # print(list(graph.edges))
