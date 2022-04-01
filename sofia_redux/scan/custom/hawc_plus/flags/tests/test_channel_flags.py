# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.hawc_plus.flags.channel_flags import \
    HawcPlusChannelFlags


def test_channel_flags():
    flags = HawcPlusChannelFlags
    for letter in HawcPlusChannelFlags.letters.keys():
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
