# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.custom.sofia.flags.quality_flags import QualityFlags


def test_channel_flags():
    flags = QualityFlags
    all_letters = 'fptunomc'
    for letter in all_letters:
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
