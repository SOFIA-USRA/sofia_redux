# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import polynd, polyexp


@pytest.fixture
def data():
    y, x = np.mgrid[:5, :5]
    # define a plane
    # This should be pretty easy to distinguish what goes where
    # z = 0 + (0.1 * x) + (0.01 * y) + (0.0001 * xy) + (0.000001 * x^2)

    z = np.full((5, 5), 1e-7)
    z += x * 0.1
    z += y * 0.01
    z += 0.0001 * x * y
    z += 0.000001 * x ** 2

    coeffs = np.zeros((3, 2))
    coeffs[0, 0] = 1e-7
    coeffs[1, 0] = 0.1
    coeffs[0, 1] = 0.01
    coeffs[1, 1] = 0.0001
    coeffs[2, 0] = 0.000001
    return np.array([x, y]), np.array(coeffs), z


def test_failure(data):
    v, c, expected = data
    with pytest.raises(ValueError):
        polynd(np.arange(2), c)

    with pytest.raises(ValueError):
        polynd(v, np.zeros((2, 2, 2)))

    with pytest.raises(ValueError):
        polynd(v, c, exponents=np.arange(10))

    with pytest.raises(ValueError):
        polynd(v, c, exponents=np.zeros((10, 1)))

    with pytest.raises(ValueError):
        polynd(v, c, covariance=np.arange(10))


def test_success(data):
    v, c, expected = data
    assert np.allclose(polynd(v, c), expected)
    e = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    c = [1e-7, 0.1, 1e-6, 0.01, 0.0001, 0]
    assert np.allclose(polynd(v, c, exponents=e), expected)


def test_covariance(data):
    v, c, expected = data
    coeffs = [[1, 1], [1, 1]]
    covar = [[0.1, 1, 0, 0],  # 0.1 + y
             [0, 0, 1, 0],  # + xy
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
    zfit, vfit = polynd(v, coeffs, covariance=covar)
    assert np.allclose(vfit, 0.1 + v[0] + (v[0] * v[1]))
    zfit, vfit = polynd(v, coeffs, covariance=np.array(covar) * np.nan)
    assert np.isnan(vfit).all()


def test_info(data):
    v, c, expected = data
    info = {}
    polynd(v, c, info=info)
    assert info['exponents'].shape == (6, 2)
    assert info['product'].shape == (6, 5, 5)
    result = polynd(v, c, info=info)
    assert np.allclose(result, expected)
    exponents = polyexp([2, 1])
    result = polynd(v, c, exponents=exponents, info=info)
    assert np.allclose(result, expected)
    result = polynd(v, c, exponents=None, product=info['product'],
                    info=info)
    assert np.allclose(result, expected)
