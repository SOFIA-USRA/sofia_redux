# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.flags import Flags
import enum

__all__ = ['ArrayFlags']


class ArrayFlags(Flags):

    class ArrayFlagTypes(enum.Flag):
        DISCARD = enum.auto()
        MASK = enum.auto()
        DEFAULT = DISCARD

    flags = ArrayFlagTypes

    descriptions = {
        flags.DISCARD: 'Discarded',
        flags.MASK: 'Masked'
    }

    letters = {
        'X': flags.DISCARD,
        'M': flags.MASK
    }
