"""
this file will be replaced by a generated one
"""

from ._stationeers import DeviceBase
from ._stationeers import FieldDesc as _FieldDesc
from ._stationeers import mk_field as _mk_field


class _Lock:
    Lock: _FieldDesc[bool] = _mk_field(bool)


class _Open:
    Open: _FieldDesc[bool] = _mk_field(bool)


class Autolathe(DeviceBase, _Lock, _Open):
    pass


class AutomatedOven(DeviceBase, _Lock):
    pass
