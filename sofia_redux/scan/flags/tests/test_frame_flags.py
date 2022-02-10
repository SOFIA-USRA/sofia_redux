# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.frame_flags import FrameFlags


def test_frame_flags():
    flags = FrameFlags
    all_letters = 'nsfJ$MWLRT<>BpkPr'
    for letter in all_letters:
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter

    f = flags.flags
    assert f.CHOP_LEFT & f.CHOP_FLAGS
    assert f.CHOP_RIGHT & f.CHOP_FLAGS
    assert f.CHOP_TRANSIT & f.CHOP_FLAGS
    assert f.FLAG_SPIKY & f.BAD_DATA
    assert f.FLAG_JUMP & f.BAD_DATA
    assert f.SKIP_MODELING & f.MODELING_FLAGS
    assert f.FLAG_DOF & f.MODELING_FLAGS
    assert f.FLAG_WEIGHT & f.MODELING_FLAGS
    assert f.FLAG_SPIKY & f.MODELING_FLAGS
    assert f.FLAG_JUMP & f.MODELING_FLAGS
    assert f.SKIP_SOURCE_MODELING & f.SOURCE_FLAGS
    assert f.MODELING_FLAGS & f.SOURCE_FLAGS
    assert f.TIME_WEIGHTING_FLAGS & f.SKIP_WEIGHTING
    assert f.TIME_WEIGHTING_FLAGS & f.MODELING_FLAGS
    assert not f.FLAG_WEIGHT & f.TIME_WEIGHTING_FLAGS
    assert not f.FLAG_DOF & f.TIME_WEIGHTING_FLAGS
