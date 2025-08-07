# pyright: reportUnusedImport=false


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

from sicc import *
from sicc import functions as f
from sicc._api import State
from sicc._api import trace_to_subr
from sicc._api import undef
from sicc._api import while_
from sicc._core import FORMAT_ANNOTATE
from sicc._core import AlwaysUnpack
from sicc._core import MVar
from sicc._core import NeverUnpack
from sicc._diagnostic import check_must_use
from sicc._diagnostic import show_pending_diagnostics
from sicc._instructions import AddF
from sicc._instructions import PredLT
from sicc._stationeers import Autolathe
from sicc._stationeers import BatchMode
from sicc._stationeers import Device
from sicc._stationeers import DeviceBase
from sicc._theme import theme
from sicc._tracing import ensure_label
from sicc._tracing import label
from sicc._tracing import trace_bundle
from sicc._tracing import trace_if
from sicc._tracing import trace_program
from sicc._transforms import regalloc_and_lower
from sicc._transforms.control_flow import build_control_flow_graph
from sicc._transforms.control_flow import compute_label_provenance
from sicc._transforms.regalloc import compute_lifetimes_all
from sicc._transforms.regalloc import regalloc
from sicc._transforms.regalloc import regalloc_try_fuse


@subr()
def error():
    exit_program()


@subr()
def check_device(d: Autolathe):
    with if_(~d["On", BatchMode.AVG]):
        error()
    with if_(~d.Open.avg):
        error()
    with if_(~d["Lock"].avg):
        error()
    with if_(~d.On.avg):
        error()


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


@program()
def main():
    label("start")

    # d1, d2, d3, d4 = [Device("Autolathe", f"MyAutolathe{i}") for i in range(4)]
    d1, d2, d3, d4 = [Autolathe(f"MyAutolathe{i}") for i in range(4)]

    check_device(d1)
    check_device(d2)
    check_device(d3)
    check_device(d4)

    f.black_box(Autolathe().Lock.avg)

    # x = black_box(child(1).y)
    # # x = black_box(parent(1))

    with while_(lambda: True):
        with if_(f.black_box(True)):
            break_()

    # parent(1)

    # # with trace_bundle():

    # with if_(black_box(x > 2)):
    #     black_box(parent(x))

    # black_box(55)

    jump("start")


if __name__ == "__main__":
    main.cli()
