# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils \
    import solve_inverse_covariance_matrices

import numpy as np


def test_solve_covariance_matrices():
    # Just exercise options, not values which are in relevant unit tests

    rand = np.random.RandomState(0)
    phi = rand.random((2, 100))
    error = rand.random(100)
    residuals = rand.random(100)
    weights = rand.random(100)

    e_cov, r_cov = solve_inverse_covariance_matrices(
        phi, error, residuals, weights,
        calculate_residual=False, calculate_error=False)
    assert e_cov.shape == (0, 0)
    assert r_cov.shape == (0, 0)

    e_cov_real, r_cov = solve_inverse_covariance_matrices(
        phi, error, residuals, weights,
        calculate_residual=False,
        calculate_error=True)

    assert np.allclose(e_cov_real, r_cov)

    e_cov, r_cov_real = solve_inverse_covariance_matrices(
        phi, error, residuals, weights,
        calculate_residual=True,
        calculate_error=False)

    assert not np.allclose(e_cov_real, r_cov_real)
    assert np.allclose(r_cov_real, e_cov)

    e_cov, r_cov = solve_inverse_covariance_matrices(
        phi, error, residuals, weights,
        calculate_residual=True,
        calculate_error=True)

    assert np.allclose(e_cov, e_cov_real)
    assert np.allclose(r_cov, r_cov_real)

    e_cov, r_cov = solve_inverse_covariance_matrices(
        phi, error, residuals, weights,
        calculate_error=True,
        calculate_residual=True,
        estimate_covariance=True)

    assert not np.allclose(e_cov, e_cov_real)
    assert np.allclose(r_cov, r_cov_real)

    amat = rand.random((2, 2))
    e_cov, r_cov = solve_inverse_covariance_matrices(
        phi, error, residuals, weights,
        error_weighted_amat=amat,
        calculate_error=True,
        calculate_residual=True,
        estimate_covariance=False)

    assert not np.allclose(e_cov, e_cov_real)
