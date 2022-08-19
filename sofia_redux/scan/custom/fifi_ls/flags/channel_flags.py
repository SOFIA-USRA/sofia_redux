# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.channel_flags import ChannelFlags
import enum

__all__ = ['FifiLsChannelFlags']


class FifiLsChannelFlags(ChannelFlags):

    class FifiLsChannelFlagTypes(enum.Flag):
        DEAD = enum.auto()
        BLIND = enum.auto()
        DISCARD = enum.auto()
        GAIN = enum.auto()
        SENSITIVITY = enum.auto()
        DOF = enum.auto()
        SPIKY = enum.auto()
        DAC_RANGE = enum.auto()
        PHASE_DOF = enum.auto()
        TIME_WEIGHTING = enum.auto()
        ROW = enum.auto()
        COL = enum.auto()
        SPEXEL = enum.auto()
        SPAXEL = enum.auto()

    flags = FifiLsChannelFlagTypes

    descriptions = {
        flags.DEAD: 'Dead',
        flags.BLIND: 'Blind',
        flags.DISCARD: 'Discarded',
        flags.GAIN: 'Gain',
        flags.SENSITIVITY: 'Noisy',
        flags.DOF: 'Degrees-of-freedom',
        flags.SPIKY: 'Spiky',
        flags.DAC_RANGE: 'Railing/Saturated',
        flags.PHASE_DOF: 'Insufficient phase degrees-of-freedom',
        flags.TIME_WEIGHTING: 'Time weighting',
        flags.ROW: 'Bad detector row gain',
        flags.COL: 'Bad detector col gain',
        flags.SPEXEL: 'Bad spexel gain',
        flags.SPAXEL: 'Bad spaxel gain',
    }

    letters = {
        'X': flags.DEAD,
        'B': flags.BLIND,
        'd': flags.DISCARD,
        'g': flags.GAIN,
        'n': flags.SENSITIVITY,
        'f': flags.DOF,
        's': flags.SPIKY,
        'r': flags.DAC_RANGE,
        'F': flags.PHASE_DOF,
        't': flags.TIME_WEIGHTING,
        'R': flags.ROW,
        'c': flags.COL,
        'e': flags.SPEXEL,
        'a': flags.SPAXEL,
    }
