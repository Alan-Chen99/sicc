from pathlib import Path

from rich import print, reconfigure
from rich.panel import Panel
from rich.pretty import Pretty, pretty_repr
from rich.theme import Theme
from rich.themes import DEFAULT

from stationeers_ic10._core import MVar
from stationeers_ic10._instructions import AddF, PredLT
from stationeers_ic10._stationeers import BatchMode
from stationeers_ic10._theme import theme
from stationeers_ic10._tracing import if_, label, trace_main_test
from stationeers_ic10.functions import black_box, jump


def main():
    reconfigure(theme=theme)

    with trace_main_test() as res:
        start = label("start")
        x = black_box(1) + black_box(True) + False
        y = x + 1.0
        z = black_box(start)

        with if_(y < 5):
            black_box(x + y)
            black_box("test")
            black_box(BatchMode.MEAN)
            jump(z)

        black_box(5)

    return res.value


if __name__ == "__main__":
    ans = main()
    # for b in ans.blocks.values():
    #     for x in b.contents:
    #         print(x)

    print(ans)
