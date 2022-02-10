# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.flags import Flags
import enum

__all__ = ['QualityFlags']


class QualityFlags(Flags):

    class QualityFlagTypes(enum.Flag):
        FAIL = enum.auto()
        PROBLEM = enum.auto()
        TEST = enum.auto()
        USABLE = enum.auto()
        NOMINAL = enum.auto()
        ORIGINAL = enum.auto()
        MODIFIED = enum.auto()
        CORRECTED = enum.auto()

    flags = QualityFlagTypes

    default_quality = flags.NOMINAL

    descriptions = {
        flags(0): "Unknown",
        flags.FAIL: 'Failed',
        flags.PROBLEM: 'Problem',
        flags.TEST: 'Test',
        flags.USABLE: 'Usable',
        flags.NOMINAL: 'Nominal',
        flags.ORIGINAL: 'Original',
        flags.MODIFIED: 'Modified',
        flags.CORRECTED: 'Corrected'
    }

    letters = {
        'f': flags.FAIL,
        'p': flags.PROBLEM,
        't': flags.TEST,
        'u': flags.USABLE,
        'n': flags.NOMINAL,
        'o': flags.ORIGINAL,
        'm': flags.MODIFIED,
        'c': flags.CORRECTED
    }
