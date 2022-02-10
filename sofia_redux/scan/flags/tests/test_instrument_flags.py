# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.instrument_flags import InstrumentFlags


def test_instrument_flags():
    flags = InstrumentFlags
    all_letters = 'sb'
    for letter in all_letters:
        assert flags.flag_to_letter(flags.letter_to_flag(letter)) == letter
