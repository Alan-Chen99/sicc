"""
this file will be replaced by a generated one

this is inspired by
https://github.com/aproposmath/stationeers-pytrapic/blob/main/src/stationeers_pytrapic/structures_generated.py
"""

from ._stationeers import DeviceTyped
from ._stationeers import FieldDesc as _FieldDesc
from ._stationeers import mk_field as _mk_field


class _On:
    On: _FieldDesc[bool] = _mk_field(bool)


class _Lock:
    Lock: _FieldDesc[bool] = _mk_field(bool)


class _Open:
    Open: _FieldDesc[bool] = _mk_field(bool)


class _Setting:
    Setting: _FieldDesc = _mk_field()


class Autolathe(DeviceTyped, _Lock, _Open, _On):
    pass


class AutomatedOven(DeviceTyped, _Lock):
    pass


class AdvancedFurnace(DeviceTyped, _Setting):
    pass
