from sicc import *
from sicc.devices import *

# subr args and returns can be:
# (1) a primitive (int/float/etc)
# (2) a tuple, list, dict; possibly nested
# (3) a dataclass; this "dataclass" is from optree,
#     its different from dataclasses.dataclass, but shares the same interface


@dataclass
class Group:
    x: AdvancedFurnace
    y: VolumePump

    # methods work too
    @subr
    def do_something(self, arg: Float) -> tuple[Float, Float]:
        self.y.Setting = arg
        return self.x.TotalMoles, self.x.TotalMoles + arg


@subr
def do_other(g: Group, arg: Float) -> Float:
    a, b = g.do_something(arg)
    return g.do_something(a + b)[0]


@program
def main():
    group1 = Group(AdvancedFurnace("A"), VolumePump("B"))
    group2 = Group(AdvancedFurnace("C"), VolumePump("D"))

    x = do_other(group1, 1)
    do_other(group2, x)


if __name__ == "__main__":
    main.cli()
