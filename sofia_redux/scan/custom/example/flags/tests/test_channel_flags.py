# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.example.flags.channel_flags import \
    ExampleChannelFlags


def test_channel_flags():
    flags = ExampleChannelFlags
    all_letters = 'XBdgnfsrFtbmRT'
    for letter in all_letters:
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
