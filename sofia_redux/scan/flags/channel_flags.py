# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.flags import Flags
import enum

__all__ = ['ChannelFlags']


class ChannelFlags(Flags):

    class ChannelFlagTypes(enum.Flag):
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

    flags = ChannelFlagTypes

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
        flags.TIME_WEIGHTING: 'Time weighting'
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
        't': flags.TIME_WEIGHTING
    }

    @classmethod
    def hardware_flags(cls):
        return cls.flags.DEAD | cls.flags.BLIND

    @classmethod
    def software_flags(cls):
        return cls.all_excluding(cls.hardware_flags())

    @classmethod
    def critical_flags(cls):
        return cls.flags.DEAD | cls.flags.BLIND | cls.flags.GAIN

    @classmethod
    def non_detector_flags(cls):
        return cls.flags.DEAD

    @classmethod
    def sourceless_flags(cls):
        return cls.flags.BLIND | cls.flags.DEAD | cls.flags.DISCARD

    @classmethod
    def source_flags(cls):
        return cls.all_excluding(cls.sourceless_flags())
