# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.example.flags.frame_flags import \
    ExampleFrameFlags


def test_channel_flags():
    flags = ExampleFrameFlags
    all_letters = 'nsfJ$MWLRT<>BpkPjNr'
    for letter in all_letters:
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
