# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import poly1d


@pytest.fixture
def data():
    coeffs = [0, 1.0, 0.5, 0.3]
    x = np.linspace(-10, 10, 1000)
    pfit = np.poly1d(np.flip(coeffs))
    y = pfit(x)
    return x, y, np.array(coeffs)


def test_failure(data):
    x, y_expected, c = data
    covar = np.zeros((1, 1, 1, 1))
    with pytest.raises(ValueError):
        poly1d(x, c, covar=covar)


def test_expected_output(data):
    x, y_expected, c = data
    assert np.allclose(y_expected, poly1d(x, c))


def test_expected_var(data):
    coeffs = [1, 3]  # y = 3x + 1
    covar = [0.1, 0.5]
    x = np.arange(5).astype(float)
    y, v = poly1d(x, coeffs, covar=covar)
    assert np.allclose(y, [1, 4, 7, 10, 13])
    assert np.allclose(v, [0.1, 0.6, 2.1, 4.6, 8.1])
    covar = [[0.1, 0.05],  # 2D covariance matrix
             [0.05, 0.5]]
    y, v = poly1d(x, coeffs, covar=covar)
    assert np.allclose(y, [1, 4, 7, 10, 13])
    assert np.allclose(v, [0.1, 0.7, 2.3, 4.9, 8.5])

    # single valued
    y, v = poly1d(1, coeffs, covar=covar)
    assert y == 4 and v == 0.7
