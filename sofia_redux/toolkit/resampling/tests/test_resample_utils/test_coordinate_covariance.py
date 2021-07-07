# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import coordinate_covariance

import numpy as np


def test_default_coordinate_covariance():
    coordinates = np.random.random((3, 100))
    covariance = coordinate_covariance(coordinates)

    mean = np.mean(coordinates, axis=1)
    dx = coordinates - mean[:, None]

    expected_covariance = dx @ dx.T / 99
    assert np.allclose(covariance, expected_covariance)


def test_mask():
    coordinates = np.random.random((3, 100))
    mask = np.full(100, True)
    mask[50:] = False
    covariance = coordinate_covariance(coordinates, mask=mask)

    coordinates = coordinates[:, mask]
    mean = np.mean(coordinates, axis=1)
    dx = coordinates - mean[:, None]
    expected_covariance = dx @ dx.T / 49
    assert np.allclose(covariance, expected_covariance)


def test_mean():
    coordinates = np.random.random((3, 100))
    mean = np.zeros(3)
    covariance = coordinate_covariance(coordinates, mean=mean)
    expected_covariance = coordinates @ coordinates.T / 99
    assert np.allclose(covariance, expected_covariance)


def test_dof():
    coordinates = np.random.random((3, 100))
    covariance = coordinate_covariance(coordinates, dof=10)
    mean = np.mean(coordinates, axis=1)
    dx = coordinates - mean[:, None]
    sum_dx2 = dx @ dx.T

    assert np.allclose(sum_dx2 / 90, covariance)

    covariance = coordinate_covariance(coordinates, dof=0)
    assert np.allclose(sum_dx2 / 100, covariance)
