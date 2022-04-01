# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.hawc_plus.flags.frame_flags import \
    HawcPlusFrameFlags


def test_channel_flags():
    flags = HawcPlusFrameFlags
    for letter in flags.letters.keys():
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
