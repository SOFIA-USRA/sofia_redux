# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import sincinterp


@pytest.fixture
def deltafunc():
    x = np.arange(10)
    y = np.zeros(10)
    y[5] = 1
    return x, y


@pytest.fixture
def data():
    x = np.arange(10)
    y = np.arange(10)
    return x, y


def test_invalid_input(data):
    x, y = data
    with pytest.raises(ValueError) as err:
        sincinterp(x, y, 5, ksize=-20)
    assert 'ksize must be a positive' in str(err.value)
    with pytest.raises(ValueError) as err:
        sincinterp(x, y[:-1], 5)
    assert 'x and y array shape mismatch' in str(err.value)
    assert np.allclose(sincinterp(x, y * np.nan, x, cval=-1), -1)
    xbad = np.append(x[5:], x[:5])
    assert np.allclose(sincinterp(xbad, y, x, skipsort=True),
                       [0, 0, 0, 0, 0, 9, 9, 9, 9, 9])


def test_expected_output(deltafunc):
    """
    Compared directly with IDL results
    """
    x, y = deltafunc
    xout = np.arange(10) / 10 + 5
    result = sincinterp(x, y, xout)
    expected_results = [1.0000000, 0.98270082, 0.93195343, 0.85111046,
                        0.74544865, 0.62172878, 0.48764449, 0.35120672,
                        0.22012204, 0.10122432]
    assert np.allclose(result, expected_results, rtol=1e-3), 'IDL agrees'
    assert np.allclose(sincinterp(x, y, x), y), 'on point interpolation'


def test_options(deltafunc):
    x, y = deltafunc
    y[4:6] = 1  # step function
    xout = np.arange(10) / 10 + 5
    result1 = sincinterp(x, y, xout, dampfac=1)
    result2 = sincinterp(x, y, xout, dampfac=3)
    assert result1[-1] < result2[-1]
    result1 = sincinterp(x, y, xout)
    result2 = sincinterp(x, y, xout, ksize=1)
    assert result1[-1] < result2[-1]


def test_single_output(data):
    x, y = data
    yo = sincinterp(x, y, 1)
    assert isinstance(yo, float)
    assert yo == 1
