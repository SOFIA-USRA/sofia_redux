# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    derivative_mscp, polynomial_exponents,
    polynomial_derivative_map, polynomial_terms)

import numpy as np


def test_mean_derivative_cross_products():
    exponents = polynomial_exponents(1, ndim=2)
    rand = np.random.RandomState(0)
    coordinates = rand.random((2, 1000))
    phi = polynomial_terms(coordinates, exponents)
    derivative_map = polynomial_derivative_map(exponents)

    # z = x + 2y
    coefficients = np.array([0.0, 1.0, 2.0])
    weights = np.ones(1000)

    g2 = derivative_mscp(coefficients, phi, derivative_map,
                         weights)
    assert np.allclose(g2, [[1, 2], [2, 4]])


def test_weighting():
    exponents = polynomial_exponents(2, ndim=2)
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    coordinates = np.stack((x.ravel(), y.ravel()))
    phi = polynomial_terms(coordinates, exponents)
    derivative_map = polynomial_derivative_map(exponents)

    # z = x^2 + y^2
    coefficients = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    weights = np.ones(coordinates.shape[1])

    g2 = derivative_mscp(coefficients, phi, derivative_map,
                         weights)

    assert np.all(np.diag(g2) > 0)
    assert np.allclose(g2[0, 1], 0)

    w = weights.copy()
    w[coordinates[0] > 0] = 0  # all negative x derivatives
    w[coordinates[1] < 0] = 0  # all positive y derivatives
    g2 = derivative_mscp(coefficients, phi, derivative_map, w)
    assert np.all(np.diag(g2) > 0)
    assert g2[0, 1] < 0
