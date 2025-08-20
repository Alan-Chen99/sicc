"""
sicc equivalent of the ic11 example here:
https://github.com/Raibo/ic11/wiki
"""

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


if __name__ == "__main__":
    main.cli()
