# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial \
    import Polyfit, linear_polyfit, gaussj_polyfit, nonlinear_polyfit


@pytest.fixture
def data():
    y, x = np.mgrid[:5, :5]
    z = 1.0 + x + (x * y) + (y ** 2)
    c = np.array([[1, 1, 0], [0, 1, 1]]).ravel()
    return x, y, z, c


def test_success(data):
    x, y, z, c = data
    poly = Polyfit(x, y, z, 2)
    assert np.allclose(poly.coefficients, c)
    assert np.allclose(poly(x, y), z)
    assert np.isclose(poly(1, 1), 4)
    r, c = poly(1, 1, dovar=True)
    assert np.allclose([r, c], [4, 0.11857142857])
    assert poly.covariance.shape == (6, 6)
    assert np.allclose(poly(x, y), z)
    assert np.isclose(poly.stats.chi2, 0)
    z[2, 2] += 1  # can't fit this exactly with 2nd order polynomial
    poly = Polyfit(x, y, z, 2)
    assert not np.isclose(poly.stats.chi2, 0)


def test_get_coefficients(data):
    x, y, z, _ = data
    poly = Polyfit(x, y, z, 2)
    c = poly.get_coefficients(covar=False)
    assert np.allclose(c, [[1, 1, 0],
                           [0, 1, 0],
                           [1, 0, 0]])
    _, c = poly.get_coefficients(covar=True)
    assert c.shape == (9, 9)
    assert np.allclose(c, c.T)
    u = np.unique(c)
    assert not np.allclose(u, u[0])

    poly = Polyfit(x, y, z, [1, 2])  # max exponent of x is 1, y is 2
    poly.get_coefficients(covar=False)


def test_robust(data):
    x, y, z, c = data
    # The fit should fail on perfect data since everything
    # will be flagged as an outlier with sufficiently low
    # rejection threshold
    poly = Polyfit(x, y, z, 2, robust=1e-6)
    assert not poly.success
    assert poly.termination == "insufficient samples remain"

    # add some noise
    noise = np.random.normal(0, 1e-3, z.shape)
    z += noise

    # should only require 1 iteration
    poly = Polyfit(x, y, z, 2, robust=5)
    assert poly.termination == "delta_rms = 0"

    poly = Polyfit(x, y, z, 2, robust=-1)
    assert poly._iteration == 1

    poly = Polyfit([1, 2], [1, 2], 1, robust=-1)
    poly._iterate()
    assert poly.stats.rchi2 is np.inf
    assert np.isnan(poly.stats.q)
    assert poly.stats.dof == 0


def test_parameters_string(data):
    x, y, z, c = data
    m = Polyfit(x, y, z, 2)
    s = m._parameters_string()
    assert "(1, 1) : 1.000000 +/- 0.100000" in s
    del m.stats.sigma
    s = m._parameters_string()
    assert "+/-" not in s
    assert "(1, 1) : 1.000000" in s


def test_parse_model_args(data):
    x, y, z, c = data
    m = Polyfit(x, y, z, 2)

    with pytest.raises(ValueError) as err:
        Polyfit(x, y, z, np.ones(1))
    assert "order size does not match" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        Polyfit(x, y, z, np.ones(1), set_exponents=True)
    assert "order must have 2 features" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        Polyfit(x, y, z, np.ones((4, 3)), set_exponents=True)
    assert "dimension 1 of order does not" in str(err.value).lower()

    m = Polyfit(x, y, z, np.ones((3, 2)), set_exponents=True)
    assert m._order == -1

    with pytest.raises(ValueError) as err:
        Polyfit(x, y, z, 2, solver='foo')
    assert "unknown solver" in str(err.value).lower()

    m = Polyfit(x, y, z, 2, solver='gaussj')
    m._parse_model_args()
    assert m.fitter is gaussj_polyfit

    m = Polyfit(x, y, z, 2, solver='linear')
    m._parse_model_args()
    assert m.fitter is linear_polyfit

    m = Polyfit(x, y, z, 2, solver='nonlinear')
    m._parse_model_args()
    assert m.fitter is nonlinear_polyfit


def test_fast_error(data):
    x, y, z, c = data
    m = Polyfit(x, y, z, 2, error=np.ones_like(z))
    m._interpolated_error = None
    m._fast_error()
    assert m._interpolated_error.size == z.size
    assert np.allclose(m._interpolated_error, 1)

    m._interpolated_error = None
    m = Polyfit(x, y, z, 2, error=2.0)
    m._interpolated_error = None
    m._fast_error()
    assert m._interpolated_error.size == z.size
    assert np.allclose(m._interpolated_error, 2)


def test_refit_mask(data):
    x, y, z, c = data
    z[3:] = 1
    z[:3] = 2
    m = Polyfit(x, y, z, 1)
    m.refit_mask((z == 1).ravel(), covar=False)
    assert np.allclose(m.coefficients, [1, 0, 0])
    assert m.covariance is None
    m.refit_mask((z == 2).ravel(), covar=True)
    assert np.allclose(m.coefficients, [2, 0, 0])
    assert np.allclose(np.diag(m.covariance),
                       [3 / 10, 1 / 30, 1 / 10])

    m.refit_mask((z == 3).ravel(), covar=True)
    assert not m.success
    assert m.covariance is None


def test_refit(data):
    x, y, z, c = data
    m = Polyfit(x, y, z, 1)
    z[3:] = 1
    z[:3] = 2
    error = np.ones_like(z)
    error[z == 2] = 0
    m.refit_data(z.ravel(), error=error.ravel())
    assert np.allclose(m.coefficients, [1, 0, 0])

    m.refit_data(z.ravel(), mask=(z == 2).ravel(), error=np.ones(z.size))
    assert np.allclose(m.coefficients, [2, 0, 0])

    z.fill(3)
    m.refit_data(x, y, z, mask=np.full(z.size, True))
    assert np.allclose(m.coefficients, [3, 0, 0])
