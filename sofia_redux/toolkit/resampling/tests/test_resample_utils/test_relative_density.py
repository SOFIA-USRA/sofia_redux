# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import (
    relative_density, calculate_distance_weights)


def uniform_spheroid_coordinates(ndim, size, sigma):
    c = np.meshgrid(*([np.linspace(-1, 1, size)] * ndim))
    c = np.stack([x.ravel() for x in c])

    if not hasattr(sigma, '__len__'):
        sigma = np.array([sigma] * ndim, dtype=float)
    else:
        sigma = np.asarray(sigma, dtype=float)
    r2 = (c ** 2).sum(axis=0)
    keep = r2 <= 1
    c = c[:, keep]
    r2 = r2[keep]
    alpha = 2 * sigma ** 2
    return c, np.sqrt(r2), sigma, alpha


def test_relative_density():

    for ndim in range(1, 4):  # Test works in dimensions 1, 2, and 3
        size = int(1e6 ** (1 / ndim))
        c, r, sigma, alpha = uniform_spheroid_coordinates(ndim, size, 0.5)
        center = np.zeros(ndim)
        weight_sum = np.sum(calculate_distance_weights(c, center, alpha))

        density = relative_density(sigma, r.size, weight_sum)

        assert np.isclose(density, 1.0, atol=1e-3)


def test_density_dip():

    for ndim in range(1, 4):  # Test works in dimensions 1, 2, and 3
        size = int(1e6 ** (1 / ndim))
        c, r, sigma, alpha = uniform_spheroid_coordinates(ndim, size, 0.5)

        keep = r > 0.5
        c = c[:, keep]
        r = r[keep]
        center = np.zeros(ndim)

        weight_sum = np.sum(calculate_distance_weights(c, center, alpha))
        density = relative_density(sigma, r.size, weight_sum)

        assert density < 1


def test_density_peak():

    for ndim in range(1, 4):  # Test works in dimensions 1, 2, and 3
        size = int(1e6 ** (1 / ndim))
        c, r, sigma, alpha = uniform_spheroid_coordinates(ndim, size, 0.5)

        keep = r < 0.5
        c = c[:, keep]
        r = r[keep]
        center = np.zeros(ndim)

        weight_sum = np.sum(calculate_distance_weights(c, center, alpha))
        density = relative_density(sigma, r.size, weight_sum)

        assert density > 1


def test_max_dimensions():

    c, r, sigma, alpha = uniform_spheroid_coordinates(3, 100, 0.5)
    keep = r < 0.5
    c = c[:, keep]
    r = r[keep]
    center = np.zeros(3)
    weight_sum = np.sum(calculate_distance_weights(c, center, alpha))

    # Should just return 1 and abort calculation
    density = relative_density(sigma, r.size, weight_sum, max_dim=1)
    assert density == 1.0
