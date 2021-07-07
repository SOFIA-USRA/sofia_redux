# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.interpspec import interpspec
import pytest


@pytest.fixture
def data():
    x = np.arange(10)
    y = x + 100.0
    error = x * 0 + 2.0
    return x, y, error


def test_invalid_input(data):
    x, y, error = data
    assert interpspec(x, y, 5, error=error, cval='a') is None
    assert interpspec(x, y[:-1], 5) is None
    assert interpspec(x, y, 5, error=error[:-1]) is None


def test_expected_output(data):
    x, y, error = data
    ynan = y.copy()
    ynan[5] = np.nan
    xout = [3, 4.5, 5.5, 6.5, 7]
    yi = interpspec(x, y, xout)
    assert np.allclose(yi, [103, 104.5, 105.5, 106.5, 107])
    yi, ei = interpspec(x, ynan, xout, error=error)
    assert np.allclose(yi, [103, 104.5, 105.5, 106.5, 107])
    assert np.allclose(ei, [2, 2.12, 2.12, 2.45, 2], atol=0.01)
    yi, ei = interpspec(x, ynan, xout, error=error, leavenans=True)
    assert np.allclose(yi, [103, np.nan, np.nan, 106.5, 107], equal_nan=True)
    assert np.allclose(ei, [2, 2.45, 2.45, 2.45, 2], atol=0.01)
    # Note that this works out strangely but is as expected
    # The error is maximum at exactly half way between the two interpolants.
    # When we effectively remove the point at x[5] when leavenans is False,
    # The x=4.5 is 25% of the full distance from x[4]->x[6] and x=5.5 is
    # 25% of the full distance from x[6]->x[4]
