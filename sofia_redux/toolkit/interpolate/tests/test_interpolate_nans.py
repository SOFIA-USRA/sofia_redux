# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import interpolate_nans


@pytest.fixture
def data():
    x = np.arange(100, dtype=float)
    y = x.copy()
    rand = np.random.RandomState(42)
    corrupt = rand.rand(100) > 0.3  # 70 percent corruption
    y[corrupt] = np.nan
    return x, y


def test_error(data):
    x, y = data
    result = interpolate_nans(x, y, x, order=-1)
    assert np.isnan(result).all()
    assert result.size == 100

    result = interpolate_nans(x, y, x, tck=True, order=-1)
    assert result == ([], [], [])

    result = interpolate_nans(x, y * np.nan, x)
    assert np.isnan(result).all()
    assert result.size == 100

    result = interpolate_nans(x[2:4], y[2:4], [2])
    assert result.size == 1
    assert np.isnan(result[0])

    y.fill(0)
    y[0] = np.nan
    result = interpolate_nans(x, y, np.array([0.0]))
    assert np.isnan(result).all()

    y.fill(np.nan)
    y[0] = 0
    result = interpolate_nans(x, y, x, order=1)
    assert np.isnan(result).all()


def test_expected(data):
    x, y = data
    result = interpolate_nans(x, y, x, order=1, width=10)

    # Note: these results are completely dependent on the random state
    assert np.isnan(result[:4]).all()
    assert np.allclose(result[4:], np.arange(96) + 4)

    tck = interpolate_nans(x, y, x, order=1, width=10, tck=True)
    assert len(tck) == 3


def test_loners(data):
    x, y = data
    y *= np.nan
    y[50] = 1
    y[:30] = np.arange(30)
    y[-30:] = np.arange(30)[::-1]
    result = interpolate_nans(x, y, x, order=1, width=10)
    assert np.allclose(result, y, equal_nan=True)
    result = interpolate_nans(x, y, x, order=1, width=10, tck=True)
    assert len(result) == 3
    assert result[0][0] == 50
