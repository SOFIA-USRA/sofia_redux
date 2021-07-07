# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import valid_num


def test_invalid():
    for value in ['a', 'A', None, 'a1', '1s']:
        assert not valid_num(value)


def test_valid():
    for value in [1, 1.0, '1.0', 1e2, '1e2', '1.2e2', True, False]:
        assert valid_num(value)
