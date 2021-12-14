# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import convert_to_list


def test_lists():
    result = convert_to_list(range(3))
    assert result == [0, 1, 2]
