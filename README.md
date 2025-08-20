# Stationeers IC10 Compiler (SICC)

[![CI](https://github.com/Alan-Chen99/sicc/actions/workflows/ci.yml/badge.svg)](https://github.com/Alan-Chen99/sicc/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/alan-chen99/sicc/graph/badge.svg?token=X8IE0XQ7NW)](https://codecov.io/github/alan-chen99/sicc)

SICC is yet another compiler for Stationeers IC10. It compiles to minimize code size.

This works by writing python code which gets **traced** and then compiled into IC10.
See [examples/explained.py](examples/explained.py) for how this works.

**This is a WIP; review output before using. API is subject to change without notice.**

### Features

- **Produces very optimized code, usually significantly shorter than from other tools**
- use any python features at trace time: trace time loops, if, assert, etc
- Also produce a readable equivalent of the output, which is easier to inspect manually
  - colored and with source information
  - **currently, its highly recommended to review this manually**
- operators
  - `+`, `-`, `*`, `/`
  - `&` for boolean and; `|` for boolean or, `~` for boolean not
    - this is because python `and` and `or` only work with `bool`
    - **in python, `&` and `|` have differenct precedence as `and` and `or`; use parenthesis: `(a < b) & (c < d)`**
- control flow
  - `if_`, `else_`, `while_`, `continue_`, `break_`, `return_`
    - the python counterparts would run at trace time
- math functions
  - only a small subset is supported at the moment
- Structures with pre-defined fields
  - static typing and autocomplete (with pyright only)
- embed raw asm code
  - in development
- high level construct for other operations: slots, stack, etc
  - in development
- Subroutines
  - **recursion is not supported**
  - [Nested function calls](examples/nested_calls.py)
  - SICC does not use `push` and `pop`; instead it moves `ra` to a different register if needed, saving lines
  - arguments and return value can be `tuple`, `list`, `dict`, or [classes](examples/classes.py)

## Install (more options will be supported later)

Install [uv](https://github.com/astral-sh/uv) and run

```bash
git clone https://github.com/Alan-Chen99/sicc-template.git
cd sicc-template
uv sync
# optional: upgrade to latest git version
# uv lock --upgrade
source ./.venv/bin/activate
python main.py
# or: sicc main.py
```

## Comparison with IC11

here is a equivalent of the [ic11](https://github.com/Raibo/ic11) example [here](https://github.com/Raibo/ic11/wiki):

<table>
<th>SICC</th>
<th>IC11</th>

<tr>
<td>

```python
from sicc import *
from sicc.devices import *

pa1 = d0
valve1 = d1
pa2 = d2
valve2 = d3

TargetTemp1 = 297
TargetTemp2 = 297


@program
def main():
    with loop():
        yield_()

        control_temp(pa1, valve1, TargetTemp1)
        control_temp(pa2, valve2, TargetTemp2)


@subr
def control_temp(pa: Pin, valve: Pin, targetTemp: Int):
    needCooling = pa.Temperature > targetTemp
    delta = abs(targetTemp - pa.Temperature)
    power = Variable(10)

    with if_(delta > 5):
        power.value = 100

    valve.On = needCooling
    valve.Setting = power
```

</td>
<td>

```c++
pin PA1 d0;
pin Valve1 d1;
pin PA2 d2;
pin Valve2 d3;

const TargetTemp1 = 297;
const TargetTemp2 = 297;

void Main()
{
    while(true)
    {
        yield;

        ControlTemp(0, 1, TargetTemp1);
        ControlTemp(2, 3, TargetTemp2);
    }
}

void ControlTemp(paIdx, valveIdx, targetTemp)
{
    var needCooling = Pins[paIdx].Temperature > targetTemp;
    var delta = Abs(targetTemp - Pins[paIdx].Temperature);
    var power = 10;

    if (delta > 5)
        power = 100;

    Pins[valveIdx].On = needCooling;
    Pins[valveIdx].Setting = power;
}
```

</td>
</tr>

<tr>
<th>SICC (17 lines)</th>
<th>IC11 (29 lines, excluding alias statements)</th>
</tr>

<td>

```
yield
move r1 0
move r0 1
jal 7
move r1 2
move r0 3
move ra 0
l r2 dr1 Temperature
slt r1 297 r2
sub r2 297 r2
abs r2 r2
move r3 10
ble r2 5 14
move r3 100
s dr0 On r1
s dr0 Setting r3
j ra
```

</td>
<td>

```
alias PA1 d0
alias Valve1 d1
alias PA2 d2
alias Valve2 d3
beqz 1 15
yield
push 297
push 1
push 0
jal 16
push 297
push 3
push 2
jal 16
j 4
j 9999
pop r0
pop r1
pop r2
push ra
l r3 dr0 Temperature
sgt r3 r3 r2
l r4 dr0 Temperature
sub r0 r2 r4
abs r0 r0
move r2 10
sgt r0 r0 5
beqz r0 29
move r2 100
s dr1 On r3
s dr1 Setting r2
pop ra
j ra
```

</td>
</tr>

<tr>
<th>SICC (readable equivalent; terminal output has color)</th>
</tr>

<tr>
<td>

```
     _program_start_0:  # _tracing.py:249: label(start)
     _while_body_913:  # _tracing.py:360: label(while_body)
  0: yield  # ic11_compare.py:21: yield_()
  1: move [%r1] 0
  2: move [%r0] 1
     # ic11_compare.py:23: control_temp(pa1, valve1,
     TargetTemp1)
  3: jal _[ic11_compare.control_temp]_915
  4: move [%r1] 2
  5: move [%r0] 3
  6: move [%ra] _while_body_913
     # <function control_temp at 0x7abfbfd8a200>
     _[ic11_compare.control_temp]_915:
     # ic11_compare.py:29: pa.Temperature
  7: l [%r2] %r1 Temperature
     # ic11_compare.py:29: pa.Temperature > targetTemp
  8: slt [%r1] 297 %r2
     # ic11_compare.py:30: targetTemp - pa.Temperature
  9: sub [%r2] 297 %r2
     # ic11_compare.py:30: abs(targetTemp - pa.Temperature)
 10: abs [%r2] %r2
 11: move [%r3] 10
     # ic11_compare.py:33: if_(delta > 5)
 12: ble %r2 5 _if_end_917
 13: move [%r3] 100
     _if_end_917:  # ic11_compare.py:33: if_(delta > 5)
 14: s %r0 On %r1  # ic11_compare.py:36: valve.On
 15: s %r0 Setting %r3  # ic11_compare.py:37: valve.Setting
     # jump to return address from ic11_compare.control_temp
 16: j %ra

 17: EndPlaceholder
```

</td>
</tr>

</table>

## Internals

This compiler uses Static single-assignment form (SSA), but it does not have phi instructions; instead it has mutable variables (MVar) representing varaibles that opts out of SSA invariant. MVar must be first converted from/to Var before used; The register allocator will tries its best to allocate read/write mvar operands to the same register. MVar are printed with an underline in the current main.py output.

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies; so it is recommended to use uv to install dependencies locked in `uv.lock`

This codebase is statically typed; this works with pyright only.
