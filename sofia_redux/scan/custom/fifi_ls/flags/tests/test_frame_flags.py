# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.fifi_ls.flags.frame_flags import \
    FifiLsFrameFlags


def test_channel_flags():
    flags = FifiLsFrameFlags
    for letter in flags.letters.keys():
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
