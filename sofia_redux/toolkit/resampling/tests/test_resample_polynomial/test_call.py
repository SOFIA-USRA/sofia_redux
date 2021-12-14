# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial

import numpy as np
import psutil
import pytest


@pytest.fixture
def data_2d():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    rand = np.random.RandomState(0)
    sigma = 0.05
    noise = rand.normal(loc=0, scale=sigma, size=coordinates.shape[1])
    data = np.ones(coordinates.shape[1]) + noise
    return coordinates, data, sigma


def test_orders(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error, order=2, window=3)
    fit = r(5, 5)
    assert np.isclose(fit, 1, atol=0.1)

    r = ResamplePolynomial(coordinates, data, error=error,
                           order=[2, 2], window=3)
    fit = r(5, 5)
    assert np.isclose(fit, 1, atol=0.1)


def test_return_options(data_2d):
    coordinates, data, sigma = data_2d
    r = ResamplePolynomial(coordinates, data, error=sigma, order=1, window=2.5)

    # Test singular fit
    fit, error, counts, weights, d_weights, rchi2, deriv, offset = r(
        5, 5, get_error=True, get_counts=True, get_weights=True,
        get_distance_weights=True, get_rchi2=True,
        get_cross_derivatives=True, get_offset_variance=True)

    assert np.isclose(fit, 1, atol=0.1)
    assert np.isclose(error, 0.01, atol=0.005)
    assert np.isclose(counts, 21)
    assert np.isclose(weights, 8400)
    assert np.isclose(d_weights, 21)
    assert np.isclose(rchi2, 1, atol=0.5)
    assert deriv.shape == (2, 2)
    assert np.isclose(offset, 0)

    # Test multiple fits
    fit, error, counts, weights, d_weights, rchi2, deriv, offset = r(
        [5, 6], [5, 6], get_error=True, get_counts=True, get_weights=True,
        get_distance_weights=True, get_rchi2=True,
        get_cross_derivatives=True, get_offset_variance=True)
    assert fit.shape == (2, 2)
    assert error.shape == (2, 2)
    assert counts.shape == (2, 2)
    assert weights.shape == (2, 2)
    assert d_weights.shape == (2, 2)
    assert rchi2.shape == (2, 2)
    assert deriv.shape == (2, 2, 2, 2)
    assert offset.shape == (2, 2)


def test_multi_set(data_2d):
    coordinates, data, sigma = data_2d
    data2 = np.stack([data.copy(), data.copy()])
    r = ResamplePolynomial(coordinates, data2, error=sigma,
                           order=1, window=2.5)

    # Test singular fit coordinate
    fit, error, counts, weights, d_weights, rchi2, deriv, offset = r(
        5, 5, get_error=True, get_counts=True, get_weights=True,
        get_distance_weights=True, get_rchi2=True,
        get_cross_derivatives=True, get_offset_variance=True)

    assert fit.shape == (2,)
    assert error.shape == (2,)
    assert counts.shape == (2,)
    assert weights.shape == (2,)
    assert rchi2.shape == (2,)
    assert deriv.shape == (2, 2, 2)
    assert offset.shape == (2,)

    # Test multiple fit coordinates
    fit, error, counts, weights, d_weights, rchi2, deriv, offset = r(
        [5, 6], [5, 6], get_error=True, get_counts=True, get_weights=True,
        get_distance_weights=True, get_rchi2=True,
        get_cross_derivatives=True, get_offset_variance=True)
    assert fit.shape == (2, 2, 2)
    assert error.shape == (2, 2, 2)
    assert weights.shape == (2, 2, 2)
    assert d_weights.shape == (2, 2, 2)
    assert rchi2.shape == (2, 2, 2)
    assert deriv.shape == (2, 2, 2, 2, 2)
    assert offset.shape == (2, 2, 2)


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_multiprocessing(data_2d):
    coordinates, data, sigma = data_2d
    r = ResamplePolynomial(coordinates, data, error=sigma, order=2, window=2.5)
    r(coordinates, jobs=-1, smoothing=0.5)  # jobs=2
    assert r.iteration == 1

    # Assert 2 iterations for adaptive, plus the one from before
    data_jmax = r(coordinates, jobs=-1, smoothing=0.5, adaptive_threshold=1.0,
                  adaptive_algorithm='scaled')  # jobs=2
    data_j1 = r(coordinates, jobs=1, smoothing=0.5, adaptive_threshold=1.0,
                adaptive_algorithm='scaled')

    assert np.allclose(data_jmax, data_j1, equal_nan=True)
    assert r.iteration == 5
