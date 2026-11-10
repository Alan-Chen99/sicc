from typing import Annotated
from typing import Self

import pytest

from sicc import *
from sicc.devices import *
from test_utils import compiler_raises
from test_utils import wrap_test

d = Autolathe()


@transpile
def _error():
    _sicc_illegal_ident = 1


def test_transpile_error():
    with pytest.raises(TranspileReservedPrefixError):
        _error()


@wrap_test
@program
def test_transpile_in_func():
    x = Variable(0)

    @transpile
    def inner(y: Int) -> Int:
        return x + y

    for i in range(3):
        x.value += i
        comment(f"v{i}:", inner(i))


@wrap_test
@program
@transpile
def test_call_later():
    comment("value:", later_function(d.ImportCount))


@transpile
def later_function(x: Int) -> Int:
    return x + 1


################################################################################
# control flow and operators


@wrap_test
@program
@transpile
def test_if_ifexp():
    if d.ExportCount > 10:
        comment("true case")
    else:
        comment("false case")

    comment("ifexp:", 1 if d.ImportCount > 0 else 2)


@wrap_test
@program
@transpile
def test_for_break_continue():
    for i in range(5):
        if i == 2:
            continue
        comment("i:", i)
        if i == 3:
            break

    comment("done")


@wrap_test
@program
@transpile
def test_bool_const():
    for i in [True, False]:
        for j in [True, False]:
            comment(i, "and", j, "->", i and j)
            comment(i, "or", j, "->", i or j)


@wrap_test
@program
@transpile
def test_ifexp_const():
    # on a python bool, only the chosen branch is traced, so the branches may differ in type
    yes: Static[bool] = True
    no: Static[bool] = False
    comment("yes:", *((1, 2) if yes else ("one",)))
    comment("no:", *((1, 2) if no else ("one",)))
    comment("yes:", d.ImportCount if yes else 2.5)


@wrap_test
@program
@transpile
def test_bool_operators():
    # Test not
    x = d.ExportCount > 10
    if not x:
        comment("not x is true")

    # Test and (short-circuit)
    if d.ExportCount > 0 and d.ImportCount > 0:
        comment("both positive")

    # Test or (short-circuit)
    if d.ExportCount > 100 or d.ImportCount > 0:
        comment("at least one")

    # Chained operators
    if d.ExportCount > 0 and d.ImportCount > 0 and d.On:
        comment("chained and")

    if not (d.ExportCount > 0 and d.ImportCount > 0):
        comment("not(and)")


def _side_effect() -> Bool:
    d.On = True
    return d.On


@wrap_test
@program
@transpile
def test_bool_short_circuit():
    if d.ExportCount > 100 and _side_effect():
        comment("side effect and")

    if d.ImportCount > 100 or _side_effect():
        comment("side effect or")


@wrap_test
@program
@transpile
def test_while_loop():
    i = 0
    while i < 5:
        comment("i:", i)
        i += 1

    comment("after while")


@wrap_test
@program
@transpile
def test_for_else():
    for i in range(3):
        if i == 1:
            break
        comment("i:", i)
    else:
        comment("for else")

    comment("after for")


@wrap_test
@program
@transpile
def test_while_else():
    i = 0
    while i < 3:
        if i == 1:
            break
        comment("i:", i)
        i += 1
    else:
        comment("while else")

    comment("after while")


@wrap_test
@program
@transpile
def test_for_else_nested():
    for i in range(2):
        comment("outer:", i)
        for j in range(2):
            comment("inner:", i, j)
            if j == 0:
                break
        else:
            comment("inner else")
    else:
        comment("outer else")

    comment("after nested")


################################################################################
# mutation and locals


@dataclass
class _Box:
    x: Int


@wrap_test
@program
@transpile
def test_mutation():
    v = 0
    # v transpile to _Place, but this becomes
    # assert isinstance(v.value, Variable)
    # so _Place is impossible to observe
    assert isinstance(v, Variable)
    assert not v._read_only
    v = 1
    comment("v:", v)

    with compiler_raises(Warnings.TypeError):
        v.value = 1  # pyright: ignore[reportAttributeAccessIssue]

    d.On = True
    d[d.On.logic_type] = True

    box = _Box(0)
    box.x = 5
    comment("box.x:", box.x)

    arr1 = [0, d.ImportCount + 1, 1]
    arr1[1] = d.ExportCount
    arr1[2] += 1
    comment("arr1:", arr1)
    a, b, c = arr1
    comment("a, b, c:", a, b, c)

    arr2 = stack_var([0, 0, 0])
    arr2[1] = 7
    comment("arr2:", arr2[1].read())

    y = 1
    del y
    comment("y:", y)  # pyright: ignore[reportUnboundVariable]

    x = (0, [0, 1, 2])
    # x[1][1] += x[0]
    comment("x:", x)


@dataclass
class Data:
    x: Float
    y: Float

    @transpile
    def some_method(self: Mutable[Self]):
        self.x += 1
        self.y += 1


# transpile should be able to forward annotations
# and this sort of forwarding should be resolvable in `@subr`
D1_T = Annotated[Data, Data(x=None, y=Arg(mutable=True))]  # pyright: ignore[reportArgumentType]


def traced_fn(d: Data, x_ro: bool | None = None, y_ro: bool | None = None) -> Data:
    assert isinstance(d, Data)
    assert isinstance(d.x, Variable)
    assert isinstance(d.y, Variable)
    if x_ro is not None:
        assert d.x._read_only == x_ro
    if y_ro is not None:
        assert d.y._read_only == y_ro

    return Data(d.y.value, d.x.value)


@transpile
def fn_mut_args(d1: D1_T, d2: Mutable[Data]):
    comment("args:", d1, d2)

    traced_fn(d1, x_ro=True, y_ro=False)
    traced_fn(d2, x_ro=False, y_ro=False)
    traced_fn(traced_fn(d1), x_ro=True, y_ro=True)

    d1.y += d1.x * 10

    with compiler_raises(Warnings.TypeError):
        d1.x += d1.y

    d3 = d1
    assert d3.x._inner is not d1.x._inner  # pyright: ignore
    assert d3.y._inner is not d1.y._inner  # pyright: ignore
    d3.x += d3.y * 10  # d3 is a local so mutable
    d2 = d1  # mutates d2, visible from caller

    # calls must return immutable/read-only values
    d2, d3 = traced_fn(d2), traced_fn(d3)

    comment("end:", d1, d2, d3)


@wrap_test
@program
@transpile
def test_mut_args():
    d1 = Data(1, 2)
    d2 = Data(3, 4)
    print(d1)  # Data(x=%s29, y=%s30)

    comment("data:", d1, d2)
    fn_mut_args(d1, d2)
    comment("output:", d1, d2)

    d1 = traced_fn(d1)

    with pytest.raises(TypeError):
        fn_mut_args(traced_fn(d1), d2)


@wrap_test
@program
@transpile
def test_bad_call_mutates_object():
    xs = [0]
    with pytest.raises(TypeError):
        xs.append(1)


@wrap_test
@program
@transpile
def test_method():
    x = Data(d.ImportCount.get(typ=int), d.ExportCount.get(typ=float))
    x.some_method()
    comment("x:", x)


################################################################################
# Static locals


@wrap_test
@program
@transpile
def test_static_accumulator():
    # a non-static list local would be a `_Place`; appending to its `.value` mutates a copy
    seen: Static[list[int]] = []
    i: Static[int]
    for i in range(3):
        seen.append(i * 10)
    comment("seen:", len(seen), seen[2], "last i:", i)  # pyright: ignore[reportPossiblyUnboundVariable]


# Annotated metadata stacks, so aliases that wrap Static stay static
_Idx = Annotated[Static[int], "loop index"]


@wrap_test
@program
@transpile
def test_static_alias():
    j: _Idx
    for j in range(2):
        comment("j:", j, "isinstance int:", isinstance(j, int))  # pyright: ignore[reportUnnecessaryIsInstance]

    # alias defined after this function: annotations are evaluated lazily at first call
    k: _LaterStatic = 3
    comment("k is int:", isinstance(k, int))  # pyright: ignore[reportUnnecessaryIsInstance]


_LaterStatic = Static[int]


@transpile(inline=True)
def _repeat_add(k: Static[int], x: Int) -> Int:
    # a static param is the caller's python int, so range(k) unrolls at trace time
    total = x
    for _ in range(k):
        total += x
    return total


@transpile
def _static_param_subr(k: Static[int]) -> Int:
    return k


@transpile
def _static_store_in_if(x: Int):
    n: Static[int] = 0
    if x > 0:
        n = 1
    comment("n:", n)


@transpile
def _static_store_in_while(x: Int):
    n: Static[int] = 0
    while x > 0:
        n += 1
    comment("n:", n)


@transpile
def _static_conflict():
    a: Static[int] = 0  # pyright: ignore[reportRedeclaration]
    a: int = 1
    comment("a:", a)


@transpile
def _static_bad_annotation():
    a: _NoSuchName = 0  # pyright: ignore[reportUndefinedVariable]
    comment("a:", a)


@wrap_test
@program
@transpile
def test_static_param():
    comment("repeat:", _repeat_add(2, d.ImportCount.get(typ=int)))

    with pytest.raises(TypeError):
        _static_param_subr(1)
    with pytest.raises(TranspileError):
        _static_store_in_if(d.ImportCount)
    with pytest.raises(TranspileError):
        _static_store_in_while(d.ImportCount)
    with pytest.raises(TranspileError):
        _static_conflict()
    with pytest.raises(TranspileError):
        _static_bad_annotation()


@wrap_test
@program
@transpile
def test_import_in_body():
    import math

    comment("floor(pi):", math.floor(math.pi))


_calls: list[int] = []


@wrap_test
@program
@transpile
def test_untouched_call():
    # references no non-static local: stays a plain call, so mutating a global is allowed
    _calls.clear()
    _calls.append(1)
    _calls.append(2)
    comment("calls:", len(_calls))
    d.On = True


@wrap_test
@program
@transpile
def test_lambda_reads_local():
    x = d.ImportCount.get(typ=int)
    comment("v:", (lambda: x + 1)())


################################################################################
# nesting


@wrap_test
@program
@transpile
def test_nested_helper():
    total = d.ImportCount.get(typ=int)
    box = _Box(0)
    scale: Static[int] = 3

    def bump(k: int) -> int:
        # trace-time python: `if` and `return` here run while tracing
        nonlocal total
        total += k
        box.x = total
        if k > 1:
            return k * scale
        return k

    i: Static[int]
    for i in range(3):
        comment("bump:", bump(i))
    comment("total:", total, "box.x:", box.x)


@wrap_test
@program
@transpile
def test_nested_transpile():
    acc = d.ImportCount.get(typ=int)

    @transpile
    def add(v: Int) -> Int:
        nonlocal acc
        if v > 0:
            acc += v
        return acc

    @transpile(inline=True)
    def twice(v: Int) -> Int:
        return v + v

    comment("add:", add(d.ExportCount.get(typ=int)))
    comment("add:", add(twice(d.ExportCount.get(typ=int))))
    comment("acc:", acc)


def _make_closure_shadow():
    y = 5

    @transpile(inline=True)
    def f(x: Int) -> Int:
        # the `y` of the lambda and of `dbl` are their own params, not the closure's `y`
        def dbl(y: Int) -> Int:
            return y * 2

        return (lambda y: y + 1)(x) + dbl(x) + y

    return f


_closure_shadow = _make_closure_shadow()


@wrap_test
@program
def test_closure_shadow():
    comment("v:", _closure_shadow(d.ImportCount))
