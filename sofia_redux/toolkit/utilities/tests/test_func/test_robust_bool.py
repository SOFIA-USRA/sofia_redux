# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.func import robust_bool


def test_false():
    for value in [0, '0', False, None, 'false', 'FALSE', 'no']:
        assert not robust_bool(value)


def test_true():
    for value in [1, '1', True, 'true', 'yes', 'y']:
        assert robust_bool(value)
