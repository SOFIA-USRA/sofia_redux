# Licensed under a 3-clause BSD style license - see LICENSE.rst

import enum
from sofia_redux.scan.flags.frame_flags import FrameFlags

__all__ = ['HawcPlusFrameFlags']


class HawcPlusFrameFlags(FrameFlags):

    class HawcPlusFrameFlagTypes(enum.Flag):
        FLAG_WEIGHT = enum.auto()
        FLAG_SPIKY = enum.auto()
        FLAG_DOF = enum.auto()
        FLAG_JUMP = enum.auto()
        SKIP_SOURCE_MODELING = enum.auto()
        SKIP_MODELING = enum.auto()
        SKIP_WEIGHTING = enum.auto()
        CHOP_LEFT = enum.auto()
        CHOP_RIGHT = enum.auto()
        CHOP_TRANSIT = enum.auto()
        NOD_LEFT = enum.auto()
        NOD_RIGHT = enum.auto()
        SAMPLE_DAC_RANGE = enum.auto()
        SAMPLE_SOURCE_BLANK = enum.auto()
        SAMPLE_SPIKE = enum.auto()
        SAMPLE_SKIP = enum.auto()
        SAMPLE_PHOTOMETRY = enum.auto()
        SAMPLE_PHI0_JUMP = enum.auto()
        SAMPLE_TRANSIENT_NOISE = enum.auto()
        TOTAL_POWER = enum.auto()

        CHOP_FLAGS = CHOP_LEFT | CHOP_RIGHT | CHOP_TRANSIT
        BAD_DATA = FLAG_SPIKY | FLAG_JUMP
        MODELING_FLAGS = SKIP_MODELING | BAD_DATA | FLAG_DOF | FLAG_WEIGHT
        SOURCE_FLAGS = SKIP_SOURCE_MODELING | MODELING_FLAGS
        CHANNEL_WEIGHTING_FLAGS = SKIP_WEIGHTING | MODELING_FLAGS
        TIME_WEIGHTING_FLAGS = ((SKIP_WEIGHTING | MODELING_FLAGS)
                                & ~(FLAG_WEIGHT | FLAG_DOF))

    flags = HawcPlusFrameFlagTypes

    descriptions = {
        flags.FLAG_WEIGHT: 'Noise level',
        flags.FLAG_SPIKY: 'Spiky',
        flags.FLAG_DOF: 'Degrees-of-freedom',
        flags.FLAG_JUMP: 'Jump',
        flags.SKIP_SOURCE_MODELING: 'Skip Source',
        flags.SKIP_MODELING: 'Skip Models',
        flags.SKIP_WEIGHTING: 'Skip Weighting',
        flags.CHOP_LEFT: 'Chop Left',
        flags.CHOP_RIGHT: 'Chop Right',
        flags.CHOP_TRANSIT: 'Chop Transit',
        flags.NOD_LEFT: 'Nod Left',
        flags.NOD_RIGHT: 'Nod Right',
        flags.SAMPLE_SOURCE_BLANK: 'Blanked',
        flags.SAMPLE_SPIKE: 'Spiky',
        flags.SAMPLE_SKIP: 'Skip',
        flags.SAMPLE_PHOTOMETRY: 'Photometry',
        flags.SAMPLE_PHI0_JUMP: 'phi0 jump',
        flags.SAMPLE_TRANSIENT_NOISE: 'transient noise',
        flags.SAMPLE_DAC_RANGE: 'DAC Range'
    }

    letters = {
        'n': flags.FLAG_WEIGHT,
        's': flags.FLAG_SPIKY,
        'f': flags.FLAG_DOF,
        'J': flags.FLAG_JUMP,

        '$': flags.SKIP_SOURCE_MODELING,
        'M': flags.SKIP_MODELING,
        'W': flags.SKIP_WEIGHTING,
        'L': flags.CHOP_LEFT,
        'R': flags.CHOP_RIGHT,
        'T': flags.CHOP_TRANSIT,
        '<': flags.NOD_LEFT,
        '>': flags.NOD_RIGHT,

        'B': flags.SAMPLE_SOURCE_BLANK,
        'p': flags.SAMPLE_SPIKE,
        'k': flags.SAMPLE_SKIP,
        'P': flags.SAMPLE_PHOTOMETRY,
        'j': flags.SAMPLE_PHI0_JUMP,
        'N': flags.SAMPLE_TRANSIENT_NOISE,
        'r': flags.SAMPLE_DAC_RANGE
    }
