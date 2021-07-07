# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import polyfitnd


@pytest.fixture
def data():
    y, x = np.mgrid[:5, :5]
    z = 1 + x + (x * y) + (0.5 * y ** 2)
    c = [[1, 1, 0],
         [0, 1, 0],
         [0.5, 0, 0]]
    return x, y, z, c


def test_standard(data):
    x, y, z, c = data
    assert np.allclose(polyfitnd(x, y, z, 2), c)
    z[2, 2] = 1000
    assert not np.allclose(polyfitnd(x, y, z, 2), c)
    assert np.allclose(polyfitnd(x, y, z, 2, robust=5), c)
    c, v = polyfitnd(x, y, z, 2, covar=True)
    assert np.allclose(v.shape, c.size)


def test_standard_model(data):
    x, y, z, c = data
    m = polyfitnd(x, y, z, 2, model=True, covar=False)
    assert m.covariance is None
    r, v = m(x, y, dovar=True)
    assert np.allclose(r, z) and v is None
    m = polyfitnd(x, y, z, 2, model=True, covar=True)
    r, v = m(x, y, dovar=True)
    assert np.allclose(r, z) and isinstance(z, np.ndarray)


def test_set_exponents(data):
    x, y, z, c = data
    c2 = polyfitnd(x, y, z, 2, covar=False, set_exponents=True)
    assert np.allclose(c2, [1, 1, 0, 0, 1, 0.5])
    c2, cov = polyfitnd(x, y, z, 2, covar=True, set_exponents=True)
    assert np.allclose(c2, [1, 1, 0, 0, 1, 0.5])
    assert cov.shape == (c2.size, c2.size)


def test_stats(data):
    x, y, z, c = data
    result, stats = polyfitnd(x, y, z, 2, stats=True)
    assert np.allclose(result, c)
    assert np.allclose(stats.chi2, 0)
