from typing import reveal_type

from rich import print as print

from sicc import *
from sicc.devices import *


@program
def main():
    # run `python explained.py` to see the output of this file
    # run `python explained.py -v` to see internal compiler steps
    # run `python explained.py -h` to see more options

    # outputs included in this files are produced by
    # `sicc examples/explained.py --width 80`
    # or
    # `sicc examples/explained.py --width 80 --no-src-info`
    # the output is colored,
    # so one might want to read that instead of plain text copied onto here

    # the output includes a readable version, and a equivalent raw assembly version
    # only the readable version is copied here

    # copied outputs may be from an older version of this file;
    # they will generally not have the correct line number

    # one can also run
    # `python explained.py -v`
    # to see what the compiler is actually doing

    # note: ra is always a usable register,
    # and will be used in places where the compiler determines that it wont otherwise get used
    # if there is any function call, ra will be used more often;
    # otherwise, only if there is lots of variables

    ########################################
    comment("part 1: tracing")

    # sicc will run (trace) your function once; this is called "trace time"

    # what is calling this function? raise an exception to get a stack trace
    # assert False

    # python control flow will execute at "trace time"
    # this is just a regular python list
    doors = [GlassDoor(f"Door{i}") for i in range(2)]

    # inspect variables at trace time
    print("devices:", doors)
    # prints:
    # [GlassDoor('Door0'), GlassDoor('Door1')]

    for d in doors:
        # let sicc know you want an operation by *staging* it (calling it once)
        d.Open = True

        # but what actually happened in this attribute write?

        # you can try to write some garbage:
        # d.Open = object()
        # this will raise a TypeError;
        # passing in verbose flag (-v) to sicc command
        # will give you a stack trace,
        # which will tell you what actually happens in a "d.Open = ..." call

    # the compiler will only know that you ran
    # `GlassDoor("Door0").Open = True` and then
    # `GlassDoor("Door1").Open = True`
    # its not relavent that you put these devices into a list and ran a "for d in devices" loop.
    #
    # so, the above will produce:

    ##
    # 0: * part 1: tracing
    # 1: sbn 'StructureGlassDoor' 'Door0' Open True  # explained.py:57: d.Open
    # 2: sbn 'StructureGlassDoor' 'Door1' Open True  # explained.py:57: d.Open
    ##

    # the source info comes inspecting the "stack trace":
    # when you write the device, python control flow is on the "d.Open" statement

    # this is only used to display the source info;
    # the compiler does not understand that you ever used a varaible called "d"

    ########################################
    comment()
    comment("part 2: Variable")

    # value read from devices is a python object of type Variable;
    # it represents a runtime value; it has no known value at trace time
    val = doors[0].RequiredPower.sum

    # operation on a Variable will stage the corresponding runtime operation
    res = val + 1

    doors[1].Setting = res

    comment("val:", val, "res:", res)

    # this becomes:

    ##
    # 3: * part 2: Variable
    #    # explained.py:92: devices[0].RequiredPower.sum
    # 4: lbn [%r0] 'StructureGlassDoor' 'Door0' RequiredPower SUM
    # 5: add [%r1] %r0 1  # explained.py:95: val + 1
    #    # explained.py:97: devices[1].Setting
    # 6: sbn 'StructureGlassDoor' 'Door1' Setting %r1
    # 7: * val: %r0 res: %r1
    ##

    ########################################
    comment()
    comment("part 3: Variable (continued)")

    # lets look more closely at a `Variable`

    val = doors[0].RequiredPower.max
    # print the python object at trace time:
    print(f"Variable: {val}, with trace-time type {type(val)}")
    # prints:
    # Variable: %s30, with trace-time type <class 'sicc._api.Variable'>

    # if you are using the pyright type checker or language server,
    # it understands the trace-time type
    reveal_type(val)  # Type of "val" is "VarRead[int]"

    # a variable from a device read is marked readonly
    # to prevent attempting to write to the device by writing the variable.
    # (doing so will only modify the variable, not the device)

    # so, this will throw an exception:
    # val.value = 1

    # create a mutable Variable by making a Variable explicitly:
    x = Variable(val)

    # refers to the same "Variable" python object as x
    # compiler cannot distinguish x_alias and x
    x_alias = x

    reveal_type(x)  # pyright only: Type of "x" is "Variable[int]"

    comment("x:", x)
    comment("x_alias:", x_alias)

    # mutable variables can be reassigned
    x.value = doors[1].RequiredPower + 1

    comment("x:", x)
    comment("x_alias:", x_alias)

    # this creates a new variable object and assign it to the python var "x"
    # it does NOT mutate x; x still exists,
    # but now the name "x" in this current python function,
    # "x" now referes to something different
    x = x + 5.1

    comment("new x:", x)
    comment("x_alias:", x_alias)  # still points to the old x

    # one need not assign a "Variable" to a python variable
    vs = [Variable(int, doors[i].RequiredPower) for i in range(2)]
    for i, v in enumerate(vs):
        v.value += 2
        comment(f"var {i}:", v)
    comment(f"sum:", sum(vs))

    # Output:

    ##
    #  8: * part 3: Variable (continued)
    #     # explained.py:118: doors[0].RequiredPower.max
    #  9: lbn [%r0] 'StructureGlassDoor' 'Door0' RequiredPower MAX
    # 10: * x: %r0
    # 11: * x_alias: %r0
    #     # explained.py:148: doors[1].RequiredPower
    # 12: lbn [%r0] 'StructureGlassDoor' 'Door1' RequiredPower AVG
    # 13: add [%r1] %r0 1  # explained.py:148: doors[1].RequiredPower + 1
    # 14: * x: %r1
    # 15: * x_alias: %r1
    # 16: add [%r2] %r1 5.1  # explained.py:157: x + 5.1
    # 17: * new x: %r2
    # 18: * x_alias: %r1
    #     # explained.py:163: doors[i].RequiredPower
    # 19: lbn [%r1] 'StructureGlassDoor' 'Door0' RequiredPower AVG
    # 20: add [%r1] %r1 2  # explained.py:165: v.value += 2
    # 21: * var 0: %r1
    # 22: add [%r0] %r0 2  # explained.py:165: v.value += 2
    # 23: * var 1: %r0
    # 24: add [%r0] %r1 %r0  # explained.py:167: sum(vs)
    # 25: * sum: %r0
    ##

    ########################################
    comment()
    comment("part 4: control flow")

    # once again a trace time loop
    for i, d in enumerate(doors):

        # i is a trace time constant; this string interpolation run at trace time
        comment(f"part 4, iteration {i}")

        # python "if" will run at trace time;
        if i == 0:
            comment("trace-time if")
            d.Open = False

        # special control flow must be used to condition for stuff at runtime
        with if_(d.Open == 1):
            comment("runtime if")
            d["Lock"] = True

    ##
    # 29: * part 4: control flow
    # 30: * part 4, iteration 0
    # 31: * trace-time if
    # 32: sbn 'StructureGlassDoor' 'Door0' Open False  # explained.py:210: d.Open
    # 33: lbn [%r0] 'StructureGlassDoor' 'Door0' Open AVG  # explained.py:213: d.Open
    # 34: bne %r0 1 _if_end_1325  # explained.py:213: if_(d.Open == 1)
    # 35: * runtime if
    # 36: sbn 'StructureGlassDoor' 'Door0' Lock True  # explained.py:215: d["Lock"]
    #     _if_end_1325:  # explained.py:213: if_(d.Open == 1)
    # 37: * part 4, iteration 1
    # 38: lbn [%r0] 'StructureGlassDoor' 'Door1' Open AVG  # explained.py:213: d.Open
    # 39: bne %r0 1 _program_exit_1327  # explained.py:213: if_(d.Open == 1)
    # 40: * runtime if
    # 41: sbn 'StructureGlassDoor' 'Door1' Lock True  # explained.py:215: d["Lock"]
    #     _program_exit_1327:  # _tracing.py:252: label(exit)
    ##


if __name__ == "__main__":
    main.cli()
