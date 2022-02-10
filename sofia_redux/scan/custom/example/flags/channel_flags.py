# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.channel_flags import ChannelFlags
import enum

__all__ = ['ExampleChannelFlags']


class ExampleChannelFlags(ChannelFlags):

    class ExampleChannelFlagTypes(enum.Flag):
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
        BIAS = enum.auto()
        MUX = enum.auto()
        ROW = enum.auto()
        FLICKER = enum.auto()

    flags = ExampleChannelFlagTypes

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
        flags.BIAS: 'Bad TES bias gain',
        flags.MUX: 'Bad MUX gain',
        flags.ROW: 'Bad detector row gain',
        flags.FLICKER: 'Flicker noise',
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
        'b': flags.BIAS,
        'm': flags.MUX,
        'R': flags.ROW,
        'T': flags.FLICKER,
    }
