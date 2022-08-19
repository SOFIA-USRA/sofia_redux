# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.fifi_ls.flags.channel_flags import \
    FifiLsChannelFlags


def test_channel_flags():
    flags = FifiLsChannelFlags
    for letter in FifiLsChannelFlags.letters.keys():
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
