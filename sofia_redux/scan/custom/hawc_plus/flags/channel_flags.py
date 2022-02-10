# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.channel_flags import ChannelFlags
import enum

__all__ = ['HawcPlusChannelFlags']


class HawcPlusChannelFlags(ChannelFlags):

    class HawcPlusChannelFlagTypes(enum.Flag):
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
        SUB = enum.auto()
        BIAS = enum.auto()
        MUX = enum.auto()
        ROW = enum.auto()
        SERIES_ARRAY = enum.auto()
        FLICKER = enum.auto()
        LOS_RESPONSE = enum.auto()
        ROLL_RESPONSE = enum.auto()

    flags = HawcPlusChannelFlagTypes

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
        flags.SUB: 'Bad subarray gain',
        flags.BIAS: 'Bad TES bias gain',
        flags.MUX: 'Bad MUX gain',
        flags.ROW: 'Bad detector row gain',
        flags.SERIES_ARRAY: 'Bad series array gain',
        flags.FLICKER: 'Flicker noise',
        flags.LOS_RESPONSE: 'LOS Response',
        flags.ROLL_RESPONSE: 'Roll Response'
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
        '@': flags.SUB,
        'b': flags.BIAS,
        'm': flags.MUX,
        'R': flags.ROW,
        'M': flags.SERIES_ARRAY,
        'T': flags.FLICKER,
        'L': flags.LOS_RESPONSE,
        '\\': flags.ROLL_RESPONSE
    }
