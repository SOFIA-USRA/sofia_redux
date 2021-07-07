# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    solve_fits, polynomial_exponents, polynomial_terms,
    polynomial_derivative_map, convert_to_numba_list)

import numpy as np
import pytest


def quadratic_func_2d(x, y):
    # f(x, y) = 2 - x^2 - y^2
    c = [2.0, 0.0, -1.0, 0.0, 0.0, -1.0]
    return c, c[0] + (c[2] * x ** 2) + (c[5] * y ** 2)


@pytest.fixture
def sample_data():
    x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    r = np.hypot(x, y)
    sample_mask = (r <= 1).ravel()
    sample_coordinates = np.stack([x.ravel(), y.ravel()])
    order = np.full(2, 2)
    exponents = polynomial_exponents(order)
    sample_phi_terms = polynomial_terms(sample_coordinates, exponents)
    error_level = 0.01
    rand = np.random.RandomState(0)
    noise = rand.normal(loc=0, scale=error_level, size=sample_mask.size)
    sample_error = np.full(sample_mask.size, error_level)
    coefficients, sample_data = quadratic_func_2d(x.ravel(), y.ravel())
    sample_data += noise
    derivative_term_map = polynomial_derivative_map(exponents)

    # Create a single data set
    sample_data = sample_data[None]
    sample_error = sample_error[None]
    sample_mask = sample_mask[None]

    # Create fitting points
    fit_coordinates = np.array([[-1 / 3, -1 / 3, 1 / 3, 1 / 3],
                                [-1 / 3, 1 / 3, -1 / 3, 1 / 3]])
    fit_phi_terms = polynomial_terms(fit_coordinates, exponents)

    # Distance weighting terms
    alpha = np.zeros(2)  # no standard distance weighting
    scaled_alpha = np.ones((sample_mask.size, 1, 1, 2))
    scaled_alpha[..., :] = 1
    shaped_alpha = np.zeros((sample_mask.size, 1, 2, 2))
    shaped_alpha[..., 0, 0] = 1
    shaped_alpha[..., 1, 1] = 1

    sample_indices = [np.arange(sample_mask.size) for _ in range(4)]
    sample_indices = convert_to_numba_list(sample_indices)

    return (sample_indices, sample_coordinates, sample_phi_terms, sample_data,
            sample_error, sample_mask, fit_coordinates, fit_phi_terms, alpha,
            scaled_alpha, shaped_alpha, order, noise, error_level, exponents,
            derivative_term_map, coefficients)


def test_get_options(sample_data):
    (sample_indices, sample_coordinates, sample_phi_terms, sample_data,
     sample_error, sample_mask, fit_coordinates, fit_phi_terms, alpha,
     scaled_alpha, shaped_alpha, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    adaptive_alpha = np.empty((0, 0, 0, 0))  # no adaptive weighting

    result = solve_fits(sample_indices, sample_coordinates, sample_phi_terms,
                        sample_data, sample_error, sample_mask,
                        fit_coordinates, fit_phi_terms, order, alpha,
                        adaptive_alpha,
                        get_error=False, get_rchi2=False, get_counts=False,
                        get_offset_variance=False, get_weights=False,
                        get_distance_weights=False,
                        get_cross_derivatives=False)

    assert result[0].shape == (1, 4)
    for i in range(1, len(result)):
        assert result[i].size == 0

    result = solve_fits(sample_indices, sample_coordinates, sample_phi_terms,
                        sample_data, sample_error, sample_mask,
                        fit_coordinates, fit_phi_terms, order, alpha,
                        adaptive_alpha,
                        get_error=True, get_rchi2=True, get_counts=True,
                        get_offset_variance=True, get_weights=True,
                        get_distance_weights=True, get_cross_derivatives=True)

    for x in result:
        if x.ndim == 2:
            assert x.shape == (1, 4)
        else:
            assert x.shape == (1, 4, 2, 2)

    bad_data = np.empty((0, 4))
    result = solve_fits(sample_indices, sample_coordinates, sample_phi_terms,
                        bad_data, sample_error, sample_mask,
                        fit_coordinates, fit_phi_terms, order, alpha,
                        adaptive_alpha,
                        get_error=True, get_rchi2=True, get_counts=True,
                        get_offset_variance=True, get_weights=True,
                        get_distance_weights=True, get_cross_derivatives=True)

    for x in result:
        assert x.size == 0


def test_error_shape(sample_data):
    (sample_indices, sample_coordinates, sample_phi_terms, sample_data,
     sample_error, sample_mask, fit_coordinates, fit_phi_terms, alpha,
     scaled_alpha, shaped_alpha, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    uniform_error = np.full_like(sample_error, error_level)
    single_error = uniform_error[:, :1].copy()
    adaptive_alpha = np.empty((0, 0, 0, 0))  # no adaptive weighting

    result = solve_fits(sample_indices, sample_coordinates, sample_phi_terms,
                        sample_data, uniform_error, sample_mask,
                        fit_coordinates, fit_phi_terms, order, alpha,
                        adaptive_alpha, get_error=True)

    error_u = result[1]

    result = solve_fits(sample_indices, sample_coordinates, sample_phi_terms,
                        sample_data, single_error, sample_mask,
                        fit_coordinates, fit_phi_terms, order, alpha,
                        adaptive_alpha, get_error=True)

    error_s = result[1]

    assert np.allclose(error_s, error_u)


def test_weighting(sample_data):
    (sample_indices, sample_coordinates, sample_phi_terms, sample_data,
     sample_error, sample_mask, fit_coordinates, fit_phi_terms, alpha,
     scaled_alpha, shaped_alpha, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    no_weighting = np.zeros(1)
    adaptive_alpha = np.empty((0, 0, 0, 0))  # no adaptive weighting

    no_weight_result = solve_fits(
        sample_indices, sample_coordinates, sample_phi_terms, sample_data,
        sample_error, sample_mask, fit_coordinates, fit_phi_terms, order,
        no_weighting, adaptive_alpha, get_error=True)

    standard_weighting = np.full_like(alpha, 1.0)
    standard_result = solve_fits(
        sample_indices, sample_coordinates, sample_phi_terms, sample_data,
        sample_error, sample_mask, fit_coordinates, fit_phi_terms, order,
        standard_weighting, adaptive_alpha, get_error=True)

    assert not np.allclose(standard_result[0], no_weight_result[0])

    scaled_result = solve_fits(
        sample_indices, sample_coordinates, sample_phi_terms, sample_data,
        sample_error, sample_mask, fit_coordinates, fit_phi_terms, order,
        alpha, scaled_alpha, get_error=True)

    # Since diagonal should be equal to 1
    assert np.allclose(standard_result[0], scaled_result[0])

    shaped_result = solve_fits(
        sample_indices, sample_coordinates, sample_phi_terms, sample_data,
        sample_error, sample_mask, fit_coordinates, fit_phi_terms, order,
        alpha, shaped_alpha, get_error=True)

    assert np.allclose(standard_result[0], shaped_result[0])
