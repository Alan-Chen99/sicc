from typing import Annotated
from typing import Any

from sicc import *
from sicc.devices import *
from test_utils import wrap_test

a = Autolathe()


@subr
def fn_int(x: int, y: Mutable[Variable[int]], z: Any = 1):
    comment("x, y, z:", x, y, z)
    y.value += x


@wrap_test
@program
def test_mutable_simple():
    x = Variable(int, a.ExportCount)
    fn_int(1, x, 3)
    comment("x:", x)


#####


@dataclass
class Data:
    x: Variable[int]
    y: Variable[int]


@subr
def fn_class(
    d1: Data,
    d2: Mutable[Data],
    d3: Annotated[
        Data, Arg(Data(x=None, y=Arg(mutable=True)))  # pyright: ignore[reportArgumentType]
    ],
    d4: Mutable[Annotated[Data, Arg()]],
) -> None:
    assert d1.x._read_only
    assert d1.y._read_only

    assert not d2.x._read_only
    assert not d2.y._read_only

    assert d3.x._read_only
    assert not d3.y._read_only

    assert not d4.x._read_only
    assert not d4.y._read_only

    comment("fn_class start:", d1, d2, d3, d4)

    d2.x.value += 1
    d3.y.value += d3.x.value
    d4.x.value, d4.y.value = d4.y.value, d4.x.value

    comment("fn_class end:", d1, d2, d3, d4)


@wrap_test
@program
def test_mutable_cls():
    d1, d2, d3, d4 = [
        Data(Variable(a.ImportCount + i), Variable(a.ExportCount - i)) for i in range(4)
    ]

    fn_class(d1, d2, d3, d4)

    comment("output:", d1, d2, d3, d4)


#####


@subr
def fn_type_spec(x: float, y: Annotated[Any, Arg(float)]):
    return x, y


@wrap_test
@program
def test_type_spec():
    comment("call1", fn_type_spec(1, 1))
    comment("call2", fn_type_spec(2.2, 2.2))
