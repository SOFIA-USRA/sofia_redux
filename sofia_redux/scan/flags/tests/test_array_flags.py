# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.flags.array_flags import ArrayFlags


def test_array_flags():
    names = list(ArrayFlags.flags.__dict__.keys())
    assert 'DISCARD' in names and 'MASK' in names and 'DEFAULT' in names
    assert ArrayFlags.flag_to_letter('DISCARD') == 'X'
    assert ArrayFlags.flag_to_letter('MASK') == 'M'
    assert ArrayFlags.flag_to_description('X') == 'Discarded'
    assert ArrayFlags.flag_to_description('M') == 'Masked'
    assert ArrayFlags.flag_to_description('default') == 'Discarded'
