"""
sicc equivalent of the ic11 example here:
https://github.com/Raibo/ic11/wiki
"""

from sicc import *
from sicc.devices import *

PA1 = d0
Valve1 = d1
PA2 = d2
Valve2 = d3

TargetTemp1 = 297
TargetTemp2 = 297


@program()
def Main():
    with loop():
        yield_()

        ControlTemp(PA1, Valve1, TargetTemp1)
        ControlTemp(PA2, Valve2, TargetTemp2)


@subr()
def ControlTemp(paIdx: Pin, valveIdx: Pin, targetTemp: Int):
    needCooling = paIdx.Temperature > targetTemp
    delta = abs(targetTemp - paIdx.Temperature)
    power = Variable(10)

    with if_(delta > 5):
        power.value = 100

    valveIdx.On = needCooling
    valveIdx.Setting = power


if __name__ == "__main__":
    Main.cli()
