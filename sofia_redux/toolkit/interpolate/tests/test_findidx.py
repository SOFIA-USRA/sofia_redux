# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import findidx


def test_single_in_point():
    assert findidx(1, 2) == 0
    assert findidx([], 2) == 0
    assert np.allclose(findidx(1, np.ones((2, 3))), np.zeros((2, 3)))


def test_monotonic():
    x = [1, 2, 3, 4, 5]
    assert findidx(x, 4) == 3
    x = [5, 4, 3, 2, 1]
    assert findidx(x, 4) == 1
    with pytest.raises(ValueError):
        x = [1, 0, 3, 4, 5]
        findidx(x, 4)
    with pytest.raises(ValueError):
        x = [1, np.nan, 3, 4, 5]
        findidx(x, 4)


def test_expected_values():
    x = [0, 1, 2, 3, 4, 5]
    assert findidx(x, 1.5) == 1.5, 'unexpected value'
    x = np.arange(10) * 2 + 1
    result = findidx(x, [5, 7])
    assert np.equal(result, [2, 3]).all(), 'unexpected value'
    assert isinstance(result[0], float), 'expected float type'
    result = findidx([0, 1, 2, 3, 4, 5],
                     [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                     left=np.nan, right=np.nan)
    nans = np.isnan(result)
    assert nans.sum() == 2, 'NaNs unexpected amount'
    assert (nans[0] == nans[-1]) & (nans[0] != nans[1])
