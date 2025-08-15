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


@program
def Main():
    with loop():
        yield_()

        ControlTemp(PA1, Valve1, TargetTemp1)
        ControlTemp(PA2, Valve2, TargetTemp2)


@subr
def ControlTemp(pa: Pin, valve: Pin, targetTemp: Int):
    needCooling = pa.Temperature > targetTemp
    delta = abs(targetTemp - pa.Temperature)
    power = Variable(10)

    with if_(delta > 5):
        power.value = 100

    valve.On = needCooling
    valve.Setting = power


if __name__ == "__main__":
    Main.cli()
