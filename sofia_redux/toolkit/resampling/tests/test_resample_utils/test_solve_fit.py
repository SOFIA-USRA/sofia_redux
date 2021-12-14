# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    solve_fit, polynomial_exponents, polynomial_terms,
    polynomial_derivative_map)

from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree

import numpy as np
import pytest


def quadratic_func_2d(x, y):
    # f(x, y) = 2 - x^2 - y^2
    c = [2.0, 0.0, -1.0, 0.0, 0.0, -1.0]
    return c, c[0] + (c[2] * x ** 2) + (c[5] * y ** 2)


@pytest.fixture
def sample_data():
    x, y = np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))
    r = np.hypot(x, y)
    window_mask = (r <= 1).ravel()
    window_coordinates = np.stack([x.ravel(), y.ravel()])
    order = np.full(2, 2)
    exponents = polynomial_exponents(order)
    window_phi = polynomial_terms(window_coordinates, exponents)
    error_level = 0.01
    rand = np.random.RandomState(0)
    noise = rand.normal(loc=0, scale=error_level, size=window_mask.size)
    window_error = np.full(window_mask.size, error_level)
    coefficients, window_values = quadratic_func_2d(x.ravel(), y.ravel())
    window_values += noise
    derivative_term_map = polynomial_derivative_map(exponents)

    return (window_coordinates, window_phi, window_values, window_error,
            window_mask, order, noise, error_level, exponents,
            derivative_term_map, coefficients)


def test_fit_failures(sample_data):
    (window_coordinates, window_phi, window_values, window_error,
     window_mask, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    bad_mask = window_mask & False
    fit_coordinate = np.zeros(2)  # center of distribution
    fit_phi = polynomial_terms(fit_coordinate, exponents)

    # No distance weighting (uniform)
    window_distance_weights = np.ones(window_mask.size)

    # This should fail since the mask has marked all samples as bad.
    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, bad_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan)

    assert np.isnan(result[0])

    # Now test bad order  # should require at least 9 points to work
    r = np.hypot(*window_coordinates)
    close_mask = window_mask.copy() & False
    sorted_distance = np.argsort(r)
    close_mask[sorted_distance[:5]] = True  # Set only 5 points valid

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, close_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan)

    assert np.isnan(result[0])

    # Now check it was the order that did this - 20 samples should be fine
    close_mask[sorted_distance[:20]] = True
    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, close_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan)
    assert not np.isnan(result[0])

    # Now test offset from the center of the distribution
    fit_coordinate = np.full(2, 1000.0)  # way off from center
    fit_phi = polynomial_terms(fit_coordinate, exponents)
    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan)
    assert np.isnan(result[0])

    # Now test edge threshold
    fit_coordinate = np.full(2, 0.5)
    fit_phi = polynomial_terms(fit_coordinate, exponents)
    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       edge_threshold=np.full(2, 0.4))
    assert np.isclose(result[0], 1.5, atol=1e-2)

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       edge_threshold=np.full(2, 0.9))
    assert np.isnan(result[0])


def test_variable_order(sample_data):
    (window_coordinates, window_phi, window_values, window_error,
     window_mask, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    tree = PolynomialTree(window_coordinates)
    tree.set_order(2, fix_order=False)
    tree.precalculate_phi_terms()
    term_indices = tree.term_indices
    derivative_term_indices = tree.derivative_term_indices
    window_phi = tree.phi_terms

    fit_coordinates = np.zeros((2, 1))
    fit_tree = PolynomialTree(fit_coordinates)
    fit_tree.set_order(2, fix_order=False)
    fit_tree.precalculate_phi_terms()

    fit_phi = fit_tree.phi_terms[:, 0]
    fit_coordinate = fit_coordinates[:, 0]

    # No distance weighting (uniform)
    window_distance_weights = np.ones(window_mask.size)

    # Without derivatives
    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       term_indices=term_indices)

    fit, error, count, weight, dweight, rchi2, deriv, offset = result
    assert np.isclose(fit, 2, atol=1e-2)
    assert np.isclose(count, window_mask.sum())
    assert np.isclose(weight, np.sum(1 / (window_error[window_mask] ** 2)))
    assert np.isclose(rchi2, 1, atol=5e-2)
    assert deriv.size == 0
    assert np.isclose(offset, 0)

    # With valid derivative mappings
    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       term_indices=term_indices,
                       derivative_term_map=derivative_term_map,
                       derivative_term_indices=derivative_term_indices)

    fit, error, count, weight, dweight, rchi2, deriv, offset = result
    assert np.isclose(fit, 2, atol=1e-2)
    assert np.isclose(count, window_mask.sum())
    assert np.isclose(weight, np.sum(1 / (window_error[window_mask] ** 2)))
    assert np.isclose(rchi2, 1, atol=5e-2)
    assert np.allclose(deriv, np.eye(2), atol=1e-2)

    r = np.hypot(*window_coordinates)

    # take the center(ish) coordinate and 4 equally spaced coordinates
    # to produce order 1 fit
    corners = r == np.sqrt(2)
    peak = r == 0
    pyramid_mask = corners | peak
    pure_values = window_values - noise

    result = solve_fit(window_coordinates, window_phi, pure_values,
                       window_error, pyramid_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       term_indices=term_indices)

    # A first order fit should be equal to the mean for this distribution
    mean_value = np.mean(pure_values[pyramid_mask])
    assert np.isclose(mean_value, result[0])
    assert not np.isclose(result[0], 2)


def test_mean_fit_and_covar(sample_data):
    (window_coordinates, window_phi, window_values, window_error,
     window_mask, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    fit_coordinate = np.zeros(2)  # center of distribution
    fit_phi = polynomial_terms(fit_coordinate, exponents)

    # No distance weighting (uniform)
    window_distance_weights = np.ones(window_mask.size)

    mean_value = np.mean(window_values[window_mask])

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       mean_fit=True)
    assert np.isclose(result[0], mean_value)
    weights = 1 / window_error[window_mask] ** 2
    v_sum = np.sum((window_error[window_mask] * weights) ** 2)
    w_sum = np.sum(weights) ** 2
    expected_error = np.sqrt(v_sum / w_sum)
    assert np.isclose(expected_error, result[1])

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       is_covar=True)

    v_sum = np.sum(window_values[window_mask] * (weights ** 2))
    expected_error_propagated = v_sum / w_sum
    assert np.isclose(expected_error_propagated, result[0])
    assert np.isnan(result[1])


def test_value_replacement(sample_data):
    (window_coordinates, window_phi, window_values, window_error,
     window_mask, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    window_error.fill(error_level / 1000)
    fit_coordinate = np.zeros(2)  # center of distribution
    fit_phi = polynomial_terms(fit_coordinate, exponents)

    # No distance weighting (uniform)
    window_distance_weights = np.ones(window_mask.size)

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       fit_threshold=1e-3)
    assert np.isclose(result[0], np.mean(window_values[window_mask]))

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       fit_threshold=-1e-3)
    assert np.isnan(result[0])


def test_distance_weight_sum(sample_data):
    (window_coordinates, window_phi, window_values, window_error,
     window_mask, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    fit_coordinate = np.zeros(2)  # center of distribution
    fit_phi = polynomial_terms(fit_coordinate, exponents)
    # No distance weighting (uniform)
    window_distance_weights = np.ones(window_mask.size)

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       get_distance_weights=True)

    assert np.isclose(result[4], window_mask.sum())

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       get_distance_weights=False)

    assert np.isclose(result[4], 0)


def test_offset_variance(sample_data):
    (window_coordinates, window_phi, window_values, window_error,
     window_mask, order, noise, error_level, exponents,
     derivative_term_map, coefficients) = sample_data

    fit_coordinate = np.full(2, 0.5)
    fit_phi = polynomial_terms(fit_coordinate, exponents)
    # No distance weighting (uniform)
    window_distance_weights = np.ones(window_mask.size)

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       get_offset_variance=True)
    assert np.isclose(result[7], 2, atol=1e-2)

    result = solve_fit(window_coordinates, window_phi, window_values,
                       window_error, window_mask, window_distance_weights,
                       fit_coordinate, fit_phi, order, cval=np.nan,
                       get_offset_variance=False)
    assert np.isnan(result[7])
