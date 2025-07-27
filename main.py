from pathlib import Path

from rich import print
from rich import reconfigure
from rich.panel import Panel
from rich.pretty import Pretty
from rich.pretty import pretty_repr
from rich.theme import Theme
from rich.themes import DEFAULT
from rich.traceback import install

from stationeers_ic10._api import if_
from stationeers_ic10._core import MVar
from stationeers_ic10._instructions import AddF
from stationeers_ic10._instructions import PredLT
from stationeers_ic10._stationeers import BatchMode
from stationeers_ic10._theme import theme
from stationeers_ic10._tracing import label
from stationeers_ic10._tracing import trace_if
from stationeers_ic10._tracing import trace_main_test
from stationeers_ic10.functions import black_box
from stationeers_ic10.functions import jump


def main():
    reconfigure(theme=theme)

    with trace_main_test() as res:
        x = black_box(1) + black_box(True) + False
        start = label("start")
        y = x + 1.0 + x
        z = black_box(start)

        with if_(y < 5):
            y.value = 1
            black_box(x < y)
            black_box("test")
            black_box(BatchMode.MEAN)
            jump(z)

        black_box(5)

    return res.value


if __name__ == "__main__":
    # install(show_locals=True)
    install()
    ans = main()
    # for b in ans.blocks.values():
    #     for x in b.contents:
    #         print(x)

    print(ans)
