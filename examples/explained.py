from typing import reveal_type

from sicc import *
from sicc import functions as f


@program(
    # set loop=True to auto loop main
    loop=False,
)
def main():
    # run `python explained.py` to see the output of this file
    # run `python explained.py -v` to see internal compiler steps
    # run `python explained.py -h` to see more options

    # outputs included in this files are produced by
    # `python explained.py`
    # or
    # `python explained.py --no-src-info`
    # the output is colored,
    # so one might want to read that instead of plain text copied onto here

    # one can also run
    # `python explained.py -v`
    # to see what the compiler is actually doing

    # note: if there is any function call, ra will be considered a usable register
    # and will be used in places where the compiler determines that it wont otherwise get used

    # sicc will run (trace) your function once; this is called "trace time"

    ########################################
    comment("part 1")

    # python control flow will execute at "trace time"
    devices = [Device("StructureGlassDoor", f"Door{i}") for i in range(2)]

    for d in devices:
        # let sicc know you want an operation by *staging* it (calling it once)
        d["Open"] = True

    # the compiler will only know that you ran `d["Open"] = True`
    # twice; its not relavent that you ran a "for d in devices" loop.
    #
    # so, the above will produce:

    ##
    # 0: Comment part 1
    # 1: sbn 'StructureGlassDoor' 'Door0' Open True
    # 2: sbn 'StructureGlassDoor' 'Door1' Open True
    ##

    ########################################
    comment("part 2")

    # once again a trace time loop
    for i, d in enumerate(devices):
        comment(f"part 2, iteration {i}")
        # "if" will run at trace time;
        if d.name == "Door0":
            comment("trace-time if")
            d["Open"] = False

        # special control flow must be used to condition for stuff at runtime
        with if_(d["Open"] == 1):
            comment("runtime if")
            d["Lock"] = True

    # this becomes:

    ##
    #  3: Comment part 2
    #  4: Comment part 2, iteration 0
    #  5: Comment trace-time if
    #  6: sbn 'StructureGlassDoor' 'Door0' Open False
    #  7: lbn [%r0] 'StructureGlassDoor' 'Door0' Open AVG
    #  8: bne %r0 1 _if_end_478
    #  9: Comment runtime if
    # 10: sbn 'StructureGlassDoor' 'Door0' Lock True
    #     _if_end_478:
    # 11: Comment part 2, iteration 1
    # 12: lbn [%r0] 'StructureGlassDoor' 'Door1' Open AVG
    # 13: bne %r0 1 _if_end_480
    # 14: Comment runtime if
    # 15: sbn 'StructureGlassDoor' 'Door1' Lock True
    #     _if_end_480:
    ##

    ########################################
    comment("part 3")

    # x is of type "Variable", representing a traced value
    # it is a regular python object;
    # the compiler does not know that it is assigned to "x".
    #
    # if you run a function with this "Variable" object as argument,
    # the compiler will see that you wanted to use the variable.
    x_readonly = f.add(1, 2)
    # the return of f.add is not mutable; if you want a mutable variable,
    # create one with Variable
    x = Variable(x_readonly)

    # refers to the same "Variable" python object as x
    x_alias = x

    reveal_type(x_readonly)  # pyright only: Type of "x" is "VarRead[int]"
    reveal_type(x)  # pyright only: Type of "x" is "Variable[int]"

    comment("x:", x)
    comment("x_alias:", x_alias)

    # one need not assign a "Variable" to a python variable
    vs = [f.add(1, i) for i in range(3)]
    for i, v in enumerate(vs):
        comment(f"var {i}:", v)

    # variables can be reassigned
    x.value = f.add(3, 4)

    comment("x:", x)
    comment("x_alias:", x_alias)

    # this creates a new variable object and assign it to the python var "x"
    x = x + 5.1

    comment("new x:", x)
    comment("x_alias:", x_alias)  # still points to the old x

    # Output:

    ##
    # 16: Comment part 3
    # 17: add [%ra] 1 2
    # 18: Comment x: %ra
    # 19: Comment x_alias: %ra
    # 20: add [%ra] 1 0
    # 21: add [%r0] 1 1
    # 22: add [%r1] 1 2
    # 23: Comment var 0: %ra
    # 24: Comment var 1: %r0
    # 25: Comment var 2: %r1
    # 26: add [%ra] 3 4
    # 27: Comment x: %ra
    # 28: Comment x_alias: %ra
    # 29: add [%r0] %ra 5.1
    # 30: Comment new x: %r0
    # 31: Comment x_alias: %ra
    ##

    ########################################
    comment("part 4")

    # variables are scoped:

    with if_(True):
        x = Variable(1)

    # throws an error: "use of out-of-scope variable" since x is out-of-scope here
    # y = x + 1

    # one need to first define the var outside
    x = Variable(1)

    with if_(True):
        x.value = 2

    # ok
    y = x + 1
    comment("y:", y)

    # output:

    ##
    # 32: Comment part 4
    # 33: add [%r0] 2 1
    # 34: Comment y: %r0
    ##

    ########################################
    comment("part 5")

    # python functions are directly called and traced
    # (the compiler would not know that you called this function)
    # they are effectively inlined

    my_function(Device("StructureGlassDoor", "MyDoor"))

    # this will make a subr, meaning that different invocations will share the same code
    # arguments and return types can be values, tuples/lists or Device
    # recursion is not supported
    also_my_function(Device("StructureGlassDoor", "MyDoor1"))
    res = also_my_function(Device("StructureGlassDoor", "MyDoor2"))
    comment("return vals:", *res)

    ##
    # 35: Comment part 5
    # 36: lbn [%ra] 'StructureGlassDoor' 'MyDoor' On MAX
    # 37: Comment On: %ra
    # 38: lbn [%ra] 'StructureGlassDoor' 'MyDoor' Lock SUM
    # 39: Comment Lock: %ra
    # 40: move [%r1] 'MyDoor1'
    # 41: jal _[__main__.also_my_function]_4848
    # 42: move [%r1] 'MyDoor2'
    # 43: jal _[__main__.also_my_function]_4848
    # 44: Comment return vals: %r0 %r0
    ##

    ##
    #     _[__main__.also_my_function]_4848:
    # 58: lbn [%r0] 'StructureGlassDoor' %r1 On MAX
    # 59: Comment On: %r0
    # 60: lbn [%r0] 'StructureGlassDoor' %r1 Lock SUM
    # 61: Comment Lock: %r0
    # 62: j %ra
    ##

    ########################################
    comment("part 6")

    d1 = Device("StructureGlassDoor", "MyDoor1")
    d2 = Device("StructureGlassDoor", "MyDoor2")

    # one can create labels and jumps to them

    label("mylabel")
    yield_()

    with if_(d1["On"].max == 0):
        jump("mylabel")

    with if_(d2["On"].max == 0):
        jump("mylabel")

    comment("jump in and out of loops")

    with loop():
        label("middle-of-loop")

        with if_(d1["Lock"].max == 0):
            jump("end-of-loop")

    label("end-of-loop")

    with if_(d2["Lock"].max == 0):
        jump("middle-of-loop")

    with loop():
        yield_()

    ##
    # 44: Comment part 6
    #     _mylabel_4633:
    # 45: yield
    # 46: lbn [%ra] 'StructureGlassDoor' 'MyDoor1' On MAX
    # 47: beq %ra 0 _mylabel_4633
    # 48: lbn [%ra] 'StructureGlassDoor' 'MyDoor2' On MAX
    # 49: beq %ra 0 _mylabel_4633
    # 50: Comment jump in and out of loops
    #     _middle-of-loop_4627:
    # 51: lbn [%ra] 'StructureGlassDoor' 'MyDoor1' Lock MAX
    # 52: bne %ra 0 _middle-of-loop_4627
    # 53: lbn [%ra] 'StructureGlassDoor' 'MyDoor2' Lock MAX
    # 54: beq %ra 0 _middle-of-loop_4627
    #     _while_body_4629:
    # 55: yield
    # 56: j _while_body_4629
    ##


def my_function(d: Device):
    comment("On:", d["On"].max)
    lock = d["Lock"].sum
    comment("Lock:", lock)
    return lock, lock


@subr()
def also_my_function(d: Device):
    return my_function(d)


if __name__ == "__main__":
    main.cli()
