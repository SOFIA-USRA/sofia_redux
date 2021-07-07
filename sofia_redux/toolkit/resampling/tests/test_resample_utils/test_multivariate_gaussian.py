# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import multivariate_gaussian

import numpy as np
import pytest


@pytest.fixture
def coordinates_2d():
    x, y = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 21))
    coordinates = np.stack((x.ravel(), y.ravel()))
    return coordinates


def test_multivariate_gaussian(coordinates_2d):
    coordinates = coordinates_2d
    sigma = np.eye(2)
    density = multivariate_gaussian(sigma, coordinates).reshape((21, 21))
    assert np.isclose(density[10, 10], 1)
    assert np.isclose(density[20, 20], np.exp(-1))


def test_normalize(coordinates_2d):
    coordinates = coordinates_2d
    sigma = np.eye(2)
    density = multivariate_gaussian(sigma, coordinates, normalize=True
                                    ).reshape((21, 21))
    assert np.isclose(density[10, 10], 1 / np.sqrt((2 * np.pi) ** 2))


def test_center(coordinates_2d):
    coordinates = coordinates_2d
    sigma = np.eye(2)
    density = multivariate_gaussian(sigma, coordinates, center=np.ones(2),
                                    ).reshape((21, 21))
    assert np.isclose(density[20, 20], 1)
    assert np.isclose(density[10, 10], np.exp(-1))
