# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.resample_utils import (
    polynomial_exponents, polynomial_terms)

from sofia_redux.toolkit.image.warp import (
    is_homography_transform, full_transform, estimate_polynomial_transform,
    warp_terms, warp_coordinates, warp_array_elements, warp,
    PolynomialTransform)


@pytest.fixture
def input_data():
    """
    Return test coordinates and data for warping.

    Returns
    -------
    data, source, destination : numpy.ndarray, numpy.ndarray, np.ndarray
    """
    dest = np.array([[0., 0.], [46., 11.5], [92., 23.], [0., 46.], [46., 57.5],
                     [92., 69.], [0., 92.], [46., 103.5], [92., 115.]])
    src = np.array(
        [[0, 0], [92, 23], [184, 46], [0, 92], [92, 115], [184, 138],
         [0, 184], [92, 207], [184, 230]])
    data = np.arange(100 * 100, dtype=float).reshape(100, 100)
    source = src.T.copy()
    destination = dest.T.copy()
    return data, source, destination


def test_is_homography_transform():
    assert not is_homography_transform(np.empty((2, 2)), 2)
    h = np.eye(3)
    assert not is_homography_transform(h, 2)
    h[2, 1] = 1.0
    assert is_homography_transform(h, 2)
    h = np.eye(3)
    h[0, 2] = 1.0
    assert is_homography_transform(h, 2)
    h = np.eye(3)
    h[2, 2] = 2
    assert is_homography_transform(h, 2)


def test_full_transform():
    a = np.deg2rad(30)
    transform = np.asarray(
        [[np.cos(a), -np.sin(a), 0],
         [np.sin(a), np.cos(a), 0],
         [0, 0, 1]])

    coordinates = np.stack((np.arange(5, dtype=float), np.arange(5)))
    t = full_transform(coordinates, transform)
    expected = transform[:2, :2] @ coordinates
    assert np.allclose(t, expected)

    # Now add a translation
    transform[0, 2] = 0.5
    transform[1, 2] = 1.0
    t = full_transform(coordinates, transform)
    expected += np.array([[0.5], [1]])
    assert np.allclose(t, expected)

    # Now add scaling
    transform[2, 2] = 2.0
    t = full_transform(coordinates, transform)
    expected /= 2
    assert np.allclose(t, expected)

    # Check array shapes
    coordinates = coordinates[:, :4].reshape((2, 2, 2))
    t = full_transform(coordinates, transform)

    assert np.allclose(t, expected[:, :4].reshape((2, 2, 2)))


def test_warp_terms():
    exponents = polynomial_exponents(2, ndim=2)
    coordinates = np.arange(20, dtype=float).reshape((2, 10))
    terms = polynomial_terms(coordinates, exponents)

    coefficients = np.ones((2, 6))
    coefficients[1] /= 2
    warped = warp_terms(terms, coefficients)
    assert np.allclose(
        warped,
        [[111, 146, 187, 234, 287, 346, 411, 482, 559, 642],
         [55.5, 73, 93.5, 117, 143.5, 173, 205.5, 241, 279.5, 321]])


def test_estimate_polynomial_transform(input_data):
    data, source, destination = input_data
    coefficients = estimate_polynomial_transform(
        source, destination, order=2, get_exponents=False)
    assert np.allclose(coefficients[0], [0, 0.5, 0, 0, 0, 0])
    assert np.allclose(coefficients[1], [0, 0, 0, 0.5, 0, 0])

    expected = coefficients.copy()
    coefficients, exponents = estimate_polynomial_transform(
        source, destination, order=2, get_exponents=True)
    assert np.allclose(coefficients, expected)
    assert np.allclose(exponents[:, 0], [0, 1, 2, 0, 1, 0])
    assert np.allclose(exponents[:, 1], [0, 0, 0, 1, 1, 2])


def test_warp_coordinates(input_data):
    data, source, destination = input_data
    coordinates = source.copy()
    assert np.allclose(warp_coordinates(coordinates, source, destination),
                       destination)


def test_warp_array_elements(input_data):
    data, source, destination = input_data
    c = warp_array_elements(source, destination, (3, 3))
    assert np.allclose(c[0, 0], 0)
    assert np.allclose(c[0, 1], 0.5)
    assert np.allclose(c[0, 2], 1)
    x = np.arange(3) / 2
    assert np.allclose(c[1, 0], x[None])
    c, transform = warp_array_elements(source, destination, (3, 3),
                                       get_transform=True)
    assert isinstance(transform, PolynomialTransform)


def test_warp(input_data):
    data, source, destination = input_data
    warped = warp(data, source, destination, order=2,
                  interpolation_order=3, mode='constant',
                  output_shape=None, cval=np.nan, clip=True)

    assert not np.isnan(warped).any()

    assert np.allclose(warped[10:12, 10:12],
                       [[505, 505.500109],
                        [555.010946, 555.511055]], atol=1e-6)

    warped = warp(data, source, destination, order=2,
                  interpolation_order=3, mode='edge',
                  output_shape=None, cval=np.nan, clip=True)
    assert np.allclose(warped[10:12, 10:12],
                       [[505, 505.500109],
                        [555.010946, 555.511055]], atol=1e-6)
    assert np.isfinite(warped).all()

    warped, transform = warp(data, source, destination, order=2,
                             interpolation_order=3, mode='edge',
                             output_shape=None, cval=np.nan, clip=True,
                             get_transform=True)
    assert isinstance(transform, PolynomialTransform)

    # Try to get NaN output
    d_off = destination.copy()
    d_off[1] -= 20
    warped = warp(data, source, d_off, order=2,
                  interpolation_order=3, mode='constant',
                  output_shape=None, cval=np.nan, clip=True)
    nans = np.isnan(warped)
    expected = np.full_like(warped, False)
    expected[:, :40] = True
    assert np.allclose(nans, expected)
