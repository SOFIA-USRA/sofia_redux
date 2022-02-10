# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.flags import Flags
import enum

__all__ = ['InstrumentFlags']


class InstrumentFlags(Flags):

    class InstrumentFlagTypes(enum.Flag):
        GAINS_SIGNED = enum.auto()
        GAINS_BIDIRECTIONAL = enum.auto()

    flags = InstrumentFlagTypes

    descriptions = {
        flags.GAINS_SIGNED: 'Signed',
        flags.GAINS_BIDIRECTIONAL: 'Bidirectional',
    }

    letters = {
        's': flags.GAINS_SIGNED,
        'b': flags.GAINS_BIDIRECTIONAL
    }
