# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    solve_polynomial_fit, polynomial_exponents, polynomial_terms,
    polynomial_derivative_map)


import numpy as np


def standard_testing_function(xy):
    # z = 1 - 0.5x^2 - 0.5y^2 + xy
    return 1 - (0.5 * xy[0] ** 2) - (0.5 * xy[1] ** 2) + (xy[0] * xy[1])


def data2d(testing_function=standard_testing_function):

    nx = ny = 300

    x, y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    xy = np.stack((x.ravel(), y.ravel()))
    f_xy = testing_function(xy)

    exponents = polynomial_exponents(2, ndim=2)
    phi = polynomial_terms(xy, exponents)

    xy_point = np.array([0.5, 0.5])
    phi_point = polynomial_terms(xy_point, exponents)

    # z = 1 - 0.5x^2 - 0.5y^2 + xy

    # point_expect = 1 - 0.125 - 0.125 + 0.25 (=1 at [0.5, 0.5])
    point_expect = 1.0

    rand = np.random.RandomState(0)
    noise_level = 0.05
    noise = rand.normal(loc=0, scale=noise_level, size=f_xy.size)

    derivative_map = polynomial_derivative_map(exponents)

    error = np.full(f_xy.size, noise_level)
    distance_weights = np.full(f_xy.size, 1.0)
    weight = distance_weights / (error ** 2)

    return (phi, phi_point, f_xy, error, distance_weights, weight,
            derivative_map, point_expect, noise)


def test_standard_fit():
    (phi, phi_point, data, error, distance_weights, weight,
     derivative_map, point_expect, noise) = data2d()

    no_error = np.empty(0)

    result, _, _, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, no_error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=False,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    assert np.isclose(result, point_expect, atol=0.01)


def test_variance():

    (phi, phi_point, data, error, distance_weights, weight,
     derivative_map, point_expect, noise) = data2d()

    no_error = np.empty(0)

    _, propagated_variance, _, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=True,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    _, close_fit_variance, _, _ = solve_polynomial_fit(
        phi, phi_point, data, no_error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=True,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    _, residual_variance, _, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, no_error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=True,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    _, no_variance, _, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, no_error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=False,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    assert no_variance == 0

    # These should be similar since we used a normal distribution of sigma
    # equal to the error array
    p_over_r = propagated_variance / residual_variance
    assert np.isclose(p_over_r, 1.0, atol=0.01)

    # residuals should be very close to zero - therefore, error should also be.
    zero_over_r = close_fit_variance / residual_variance
    assert np.isclose(zero_over_r, 0)


def test_calculate_rchi2():

    (phi, phi_point, data, error, distance_weights, weight,
     derivative_map, point_expect, noise) = data2d()

    no_error = np.empty(0)

    _, _, real_rchi2, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=False,
        calculate_rchi2=True, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    _, _, unknown_rchi2, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, no_error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=False,
        calculate_rchi2=True, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    _, _, no_rchi2, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, no_error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=False,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    assert np.isclose(real_rchi2, 1, atol=0.01)
    assert real_rchi2 != 1
    assert unknown_rchi2 == 1
    assert no_rchi2 == 0


def test_derivatives():

    def uniform_gradient_function(xy):
        # z = 2x + y
        # dz/dx = 2
        # dz/dy = 1
        return 2 * xy[0] + xy[1]

    (phi, phi_point, data, error, distance_weights, weight,
     derivative_map, point_expect, noise) = data2d(
        testing_function=uniform_gradient_function)

    no_error = np.empty(0)

    # No valid derivatives
    _, _, _, invalid_gradients = solve_polynomial_fit(
        phi, phi_point, data + noise, error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=False,
        calculate_rchi2=False, calculate_derivative_mscp=True,
        error_weighting=False, estimate_covariance=False)

    assert invalid_gradients.shape == (0, 0)

    _, _, _, gradients = solve_polynomial_fit(
        phi, phi_point, data + noise, no_error, distance_weights, weight,
        derivative_term_map=derivative_map, calculate_variance=False,
        calculate_rchi2=False, calculate_derivative_mscp=True,
        error_weighting=False, estimate_covariance=False)

    # gradients = [[dx*dx, dx*dy], [dy*dx, dy*dy]]
    assert np.allclose(gradients, [[4, 2], [2, 1]], atol=0.01)

    _, _, _, no_gradients = solve_polynomial_fit(
        phi, phi_point, data + noise, no_error, distance_weights, weight,
        derivative_term_map=derivative_map, calculate_variance=True,
        calculate_rchi2=True, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    assert no_gradients.shape == (6, 6)  # r_inv was passed out as dummy


def test_estimate_covariance():

    (phi, phi_point, data, error, distance_weights, weight,
     derivative_map, point_expect, noise) = data2d()

    rand = np.random.RandomState(1)
    error = abs(rand.normal(loc=0, scale=0.05, size=phi.shape[1]))

    _, real_variance, _, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=True,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    _, estimated_variance, _, _ = solve_polynomial_fit(
        phi, phi_point, data + noise, error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=True,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=True)

    assert not np.isclose(real_variance, estimated_variance)


def test_error_weighting():

    (phi, phi_point, data, error, distance_weights, weight,
     derivative_map, point_expect, noise) = data2d()

    _, v0, _, _ = solve_polynomial_fit(
        phi, phi_point, data, error, distance_weights, weight,
        derivative_term_map=None, calculate_variance=True,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=False, estimate_covariance=False)

    # Screw up error and weighting, but get error propagation from full weight
    _, v1, _, _ = solve_polynomial_fit(
        phi, phi_point, data, error * 0, distance_weights * 0, weight,
        derivative_term_map=None, calculate_variance=True,
        calculate_rchi2=False, calculate_derivative_mscp=False,
        error_weighting=True, estimate_covariance=False)

    assert np.isclose(v0, v1)
