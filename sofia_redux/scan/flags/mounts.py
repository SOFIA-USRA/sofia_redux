# Licensed under a 3-clause BSD style license - see LICENSE.rst

import enum

__all__ = ['Mount']


class Mount(enum.Enum):
    """
    The Mount class contains all available types of telescope mount.
    """
    UNKNOWN = enum.auto()
    CASSEGRAIN = enum.auto()
    GREGORIAN = enum.auto()
    PRIME_FOCUS = enum.auto()
    LEFT_NASMYTH = enum.auto()
    RIGHT_NASMYTH = enum.auto()
    NASMYTH_COROTATING = enum.auto()
