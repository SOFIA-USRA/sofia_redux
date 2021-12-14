# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial
from sofia_redux.toolkit.resampling.tree.polynomial_tree import PolynomialTree

import numpy as np
import pytest


@pytest.fixture
def data_2d():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:11, :11]])
    data = np.full(121, 1.0)
    rand = np.random.RandomState(0)
    noise = rand.normal(loc=0, scale=0.05, size=121)
    error = np.full(121, 0.05)
    data += noise
    mask = np.full(121, True)
    mask[:50] = False
    return coordinates, data, error, mask


def test_init(data_2d):
    coordinates, data, error, mask = data_2d
    r = ResamplePolynomial(coordinates, data, order=4, fix_order=False)
    assert r._order == 4
    assert not r._fix_order
    r = ResamplePolynomial(coordinates, data, order=3, fix_order=True)
    assert r._order == 3
    assert r._fix_order


def test_set_sample_tree(data_2d):
    coordinates, data, error, mask = data_2d
    r = ResamplePolynomial(coordinates, data, order=3)
    r.sample_tree = None
    r._order = 3
    r.set_sample_tree(coordinates)
    assert isinstance(r.sample_tree, PolynomialTree)
    assert r._order == 3 and isinstance(r._order, np.ndarray)
    assert isinstance(r.sample_tree.phi_terms, np.ndarray)


def test_check_order():
    func = ResamplePolynomial.check_order
    order = np.empty((2, 2))
    n_features = 2
    n_samples = 100
    with pytest.raises(ValueError) as err:
        func(order, n_features, n_samples)
    assert "Order should be a scalar or 1-D array" in str(err.value)

    with pytest.raises(ValueError) as err:
        func(np.empty(3), n_features, n_samples)
    assert "Order vector does not match number of features" in str(err.value)

    with pytest.raises(ValueError) as err:
        func(11, n_features, n_samples)
    assert "Too few data samples for order" in str(err.value)

    with pytest.raises(ValueError) as err:
        func([10, 11], n_features, n_samples)
    assert "Too few data samples for order" in str(err.value)

    order = func([3, 4], n_features, n_samples)
    assert isinstance(order, np.ndarray) and np.allclose(order, [3, 4])


def test_calculate_minimum_points(data_2d):
    coordinates, data, error, mask = data_2d
    r = ResamplePolynomial(coordinates, data, order=3)
    assert r.calculate_minimum_points() == 16
    r = ResamplePolynomial(coordinates, data, order=[4, 5])
    assert r.calculate_minimum_points() == 30


def test_pre_fit(data_2d):
    coordinates, data, error, mask = data_2d
    r = ResamplePolynomial(coordinates, data, error=error, order=3)
    settings = r.reduction_settings(adaptive_threshold=1.0, smoothing=0.5,
                                    adaptive_algorithm='scaled')

    r.pre_fit(settings, coordinates, adaptive_region_coordinates=None)
    assert r.fit_tree.order == 3
    assert settings['order_symmetry']
    assert r.fit_tree.phi_terms.shape == (10, 121)
    full_blocks = r.fit_tree.block_population.copy()
    assert not np.any(full_blocks == 0)

    r = ResamplePolynomial(coordinates, data, error=error, order=[2, 3])
    settings = r.reduction_settings(adaptive_threshold=1.0, smoothing=0.5,
                                    adaptive_algorithm='scaled')
    r.pre_fit(settings, coordinates, adaptive_region_coordinates=None)
    assert np.allclose(r.fit_tree.order, [2, 3])
    assert not settings['order_symmetry']
    assert r.fit_tree.phi_terms.shape == (9, 121)

    r = ResamplePolynomial(coordinates, data, error=error, order=3)
    settings = r.reduction_settings(adaptive_threshold=1.0, smoothing=0.5,
                                    adaptive_algorithm='scaled')

    adaptive_region_coordinates = np.arange(-3, 3), np.arange(-3, 3)
    r.pre_fit(settings, coordinates,
              adaptive_region_coordinates=adaptive_region_coordinates)
    diff = full_blocks != r.fit_tree.block_population
    assert np.any(diff)
    assert np.allclose(r.fit_tree.block_population[diff], 0)


def test_process_block(data_2d):
    coordinates, data, error, mask = data_2d
    r = ResamplePolynomial(coordinates, data, error=error, order=3)
    settings = r.reduction_settings()
    r.pre_fit(settings, coordinates)
    g = r.global_resampling_values()
    get_error = get_counts = get_weights = get_distance_weights = False
    get_rchi2 = get_cross_derivatives = get_offset_variance = False

    filename, iteration = None, 1
    args = (r.data, r.error, r.mask, r.fit_tree, r.sample_tree,
            get_error, get_counts, get_weights, get_distance_weights,
            get_rchi2, get_cross_derivatives, get_offset_variance,
            settings)

    g['args'] = args
    g['iteration'] = iteration
    g['filename'] = filename

    block_population = r.fit_tree.block_population
    hood_population = r.sample_tree.hood_population
    skip = (block_population == 0) | (hood_population == 0)
    first_block = np.nonzero(~skip)[0][0]

    result = r.process_block((filename, iteration), first_block)
    assert len(result) == 9
    fit_indices, fit, error, counts, wsum, dwsum, rchi2, deriv, var = result
    assert isinstance(fit_indices, np.ndarray) and fit_indices.size > 0
    assert isinstance(fit, np.ndarray) and fit.size > 0

    for value in [error, counts, wsum, dwsum, rchi2, deriv, var]:
        assert isinstance(value, np.ndarray) and value.size == 0


def test_call(data_2d):
    coordinates, data, error, mask = data_2d
    r = ResamplePolynomial(coordinates, data, order=3)
    fit = r(4, 4)
    assert np.isclose(fit, 0.9956241058695572)
