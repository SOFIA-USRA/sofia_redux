# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial \
    import poly1d, nonlinear_polyfit, polyexp


@pytest.fixture
def data1d():
    coeffs = [1, 1e-1, 1e-3]
    x = np.arange(10).astype(float)
    y = poly1d(x, coeffs)
    return np.array([x, y]), coeffs


@pytest.fixture
def data2d():
    y, x = np.mgrid[:5, :5]
    # define a plane
    # This should be pretty easy to distinguish what goes where
    # z = 0 + (0.1 * x) + (0.01 * y) + (0.0001 * xy) + (0.000001 * x^2)
    x, y = x.ravel(), y.ravel()
    z = 1e-7 + (0.1 * x) + (0.01 * y) + (1e-3 * x * y) + (1e-6 * x ** 2)
    return np.array([x, y, z])


def test_1d(data1d):
    samples, expected = data1d
    info = {}
    c = nonlinear_polyfit(samples, 2, info=info,
                          product=np.zeros(1),
                          mask=np.full(samples.shape[1], True))
    assert np.allclose(c, expected)
    exponents = info['exponents']
    assert info['product'] is None
    assert np.allclose(exponents, np.arange(3)[:, None])
    c = nonlinear_polyfit(samples, -1, exponents=exponents[:, 0])
    assert np.allclose(c, expected)


def test_2d(data2d):
    samples = data2d
    info = {}
    c = nonlinear_polyfit(samples, 2, info=info)
    assert np.allclose(info['exponents'], [
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [0, 2]])
    assert np.allclose(c, [1e-7, 0.1, 1e-6, 1e-2, 1e-3, 0])
    c = nonlinear_polyfit(samples, [2, 1], info=info)
    assert np.allclose(info['exponents'], [
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]])
    assert np.allclose(c, [1e-7, 0.1, 1e-6, 1e-2, 1e-3, 0])


def test_bad_values(data2d):
    samples = data2d
    c = nonlinear_polyfit(samples * np.nan, 2)
    assert np.isnan(c).all()


def test_covar1d(data1d):
    samples, expected = data1d
    c, covar = nonlinear_polyfit(samples, 2, covar=True)
    assert covar.shape == (3, 3) and np.allclose(covar, 0)


def test_covar2d(data2d):
    samples = data2d
    c, covar = nonlinear_polyfit(samples, 2, covar=True)
    assert covar.shape == (6, 6) and not np.isfinite(covar).any()


def test_ignorenans(data2d):
    samples = data2d
    error = np.full(samples.shape[1], 1.0)
    error[-1] = np.nan
    samples[0, 0] = np.nan
    c = nonlinear_polyfit(samples, 2, error=error)
    assert not np.isnan(c).any()
    c = nonlinear_polyfit(
        samples, 2, error=error, ignorenans=False)
    assert np.isnan(c).all()
    samples *= np.nan
    c = nonlinear_polyfit(samples, 2, error=error)
    assert np.isnan(c).all()


def test_errors(data1d):
    with pytest.raises(ValueError) as err:
        nonlinear_polyfit(data1d[0][0], 2)
    assert "samples must have at least 1 feature" in str(err.value).lower()

    exponents = polyexp(2, ndim=2)
    with pytest.raises(ValueError) as err:
        nonlinear_polyfit(data1d[0], 1, exponents=exponents[None, None])
    assert "exponents must be a 1-d or 2-d array" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        nonlinear_polyfit(data1d[0], 2,
                          exponents=exponents.repeat(2, axis=1))
    assert "features mismatch" in str(err.value).lower()
