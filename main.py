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

from sicc import dataclass
from sicc import dataclasses
from sicc import pytree
from sicc._api import Float
from sicc._api import Int
from sicc._api import State
from sicc._api import Variable
from sicc._api import if_
from sicc._api import return_
from sicc._api import subr
from sicc._api import trace_to_subr
from sicc._core import FORMAT_ANNOTATE
from sicc._core import AlwaysUnpack
from sicc._core import MVar
from sicc._core import NeverUnpack
from sicc._diagnostic import check_must_use
from sicc._diagnostic import show_pending_diagnostics
from sicc._instructions import AddF
from sicc._instructions import PredLT
from sicc._stationeers import BatchMode
from sicc._theme import theme
from sicc._tracing import ensure_label
from sicc._tracing import label
from sicc._tracing import trace_bundle
from sicc._tracing import trace_if
from sicc._tracing import trace_main_test
from sicc._transforms import emit_asm
from sicc._transforms.control_flow import build_control_flow_graph
from sicc._transforms.control_flow import compute_label_provenance
from sicc._transforms.regalloc import compute_lifetimes_all
from sicc._transforms.regalloc import regalloc
from sicc._transforms.regalloc import regalloc_try_fuse
from sicc.functions import black_box
from sicc.functions import jump


@dataclass
class Data:
    x: Float
    y: Float


@subr()
def child(x: Float):
    with if_(x > 0):
        return_(Data(7, x))
    return Data(x, 9)


@subr()
def parent(x: Float) -> Float:
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

        # x = black_box(child(1).y)
        x = black_box(parent(1))
        with if_(x > 1):
            parent(x + 51)

        parent(1)

        # with trace_bundle():

        with if_(black_box(x > 2)):
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

    emit_asm(ans)

    # res = compute_lifetimes_all(ans)
    # with res.with_anno():
    #     print(ans)

    # print("conflict_graph", list(res.conflict_graph.edges))

    # fuse_res = regalloc_try_fuse(ans)
    # print("fuse groups", fuse_res.groups_rev)
    # # print("post-fuse conflict graph", list(fuse_res.collapsed_conflict_graph.edges))

    # with fuse_res.with_anno():
    #     print(ans)

    # regalloc(ans)
    print(ans)
