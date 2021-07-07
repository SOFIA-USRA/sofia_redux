# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils \
    import distribution_variances

import numpy as np


def test_distribution_variances():
    rand = np.random.RandomState(0)
    covariance = np.eye(2)
    center = np.zeros(2)
    coordinates = rand.multivariate_normal(center, covariance, 100000).T
    r = np.hypot(*coordinates)
    sigma = np.sqrt(distribution_variances(coordinates))

    # Check is linear relationship with expected coefficients
    assert np.allclose(np.polyfit(r, sigma, 1), [1, 0], atol=0.01)

    covariance /= 4
    coordinates = rand.multivariate_normal(center, covariance, 100000).T
    r = np.hypot(*coordinates)
    sigma = np.sqrt(distribution_variances(coordinates))
    assert np.allclose(np.polyfit(r, sigma, 1), [2, 0], atol=0.01)


def test_mean():
    # Just test mean is passed in
    rand = np.random.RandomState(1)
    covariance = np.eye(2)
    center = np.zeros(2)
    coordinates = rand.multivariate_normal(center, covariance, 1000).T
    v = distribution_variances(coordinates, mean=None)
    v2 = distribution_variances(coordinates, mean=np.ones(2))
    assert not np.allclose(v, v2)


def test_optional_covariance():
    rand = np.random.RandomState(2)
    covariance = np.eye(2) / 4
    center = np.zeros(2)
    coordinates = rand.multivariate_normal(center, covariance, 100000).T
    r = np.hypot(*coordinates)

    # statistics derived from the actual distribution
    sigma_fit = np.sqrt(distribution_variances(coordinates))

    c_fit = np.polyfit(r, sigma_fit, 1)

    # statistics supplied
    sigma_real = np.sqrt(distribution_variances(
        coordinates, covariance=covariance))
    c_real = np.polyfit(r, sigma_real, 1)

    assert np.allclose(c_fit, [2, 0], atol=1e-2)
    assert np.allclose(c_real, [2, 0], atol=1e-4)
    assert not np.allclose(c_fit, c_real)

    sigma_inv = np.sqrt(distribution_variances(
        coordinates, sigma_inv=np.linalg.pinv(covariance)))

    assert np.allclose(sigma_inv, sigma_real)


def test_dof():
    coordinates = np.random.random((2, 100))
    v0 = distribution_variances(coordinates, dof=0)
    v50 = distribution_variances(coordinates, dof=50)
    assert np.allclose(v0, v50 * 2)
