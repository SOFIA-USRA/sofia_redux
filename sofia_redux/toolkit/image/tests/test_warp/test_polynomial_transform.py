# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.image.warp import PolynomialTransform


@pytest.fixture
def xy_coordinates():
    source = np.asarray(
        [[0, 92, 184, 0, 92, 184, 0, 92, 184],
         [0, 23, 46, 92, 115, 138, 184, 207, 230]], dtype=float)
    destination = source / 2 + 1.0
    return source, destination


@pytest.fixture
def xy_polynomial(xy_coordinates):
    source, destination = xy_coordinates
    p = PolynomialTransform(source, destination, order=2)
    return p


@pytest.fixture
def x_polynomial():
    source = np.arange(10, dtype=float)
    destination = source * 2 + 1
    p = PolynomialTransform(source, destination, order=2)
    return p


def test_init(xy_coordinates):
    p = PolynomialTransform()
    assert p.coefficients is None
    assert p.exponents is None
    assert p.inverse_coefficients is None
    assert p.order == 2
    source, destination = xy_coordinates

    p = PolynomialTransform(source, destination, order=1)
    assert p.order == 1
    assert isinstance(p.exponents, np.ndarray)
    assert np.allclose(p.exponents, [[0, 0], [1, 0], [0, 1]])
    assert isinstance(p.coefficients, np.ndarray)
    assert np.allclose(p.coefficients, [[1, 0.5, 0], [1, 0, 0.5]])
    assert isinstance(p.inverse_coefficients, np.ndarray)
    assert np.allclose(p.inverse_coefficients, [[-2, 2, 0], [-2, 0, 2]])


def test_ndim(xy_polynomial):
    assert xy_polynomial.ndim == 2
    assert PolynomialTransform().ndim == 0


def test_n_coeffs(xy_polynomial):
    assert xy_polynomial.n_coeffs == 6
    assert PolynomialTransform().n_coeffs == 0


def test_estimate_transform(xy_coordinates):
    p = PolynomialTransform()
    p.estimate_transform(None, None, 3)
    assert p.order == 3
    assert p.coefficients is None
    assert p.exponents is None
    assert p.inverse_coefficients is None
    source, destination = xy_coordinates
    p.estimate_transform(source, destination, 2)
    assert p.order == 2
    assert isinstance(p.coefficients, np.ndarray)
    assert isinstance(p.exponents, np.ndarray)
    assert isinstance(p.inverse_coefficients, np.ndarray)
    assert np.allclose(p.coefficients,
                       [[1, 0.5, 0, 0, 0, 0],
                        [1, 0, 0, 0.5, 0, 0]])
    assert np.allclose(p.exponents,
                       [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [0, 2]])
    assert np.allclose(p.inverse_coefficients,
                       [[-2, 2, 0, 0, 0, 0],
                        [-2, 0, 0, 2, 0, 0]])

    with pytest.raises(ValueError) as err:
        p.estimate_transform(np.zeros((2, 2)), np.zeros((3, 3)))
    assert 'Source and destination coordinates' in str(err.value)


def test_transform(xy_coordinates, xy_polynomial, x_polynomial):
    source, destination = xy_coordinates
    p = PolynomialTransform()
    with pytest.raises(ValueError) as err:
        _ = p.transform(source)
    assert "No polynomial fit" in str(err.value)

    p = xy_polynomial
    with pytest.raises(ValueError) as err:
        _ = p.transform(1)
    assert 'Incompatible input dimensions' in str(err.value)

    with pytest.raises(ValueError) as err:
        _ = p.transform(np.arange(3))
    assert 'Incompatible input dimensions' in str(err.value)

    with pytest.raises(ValueError) as err:
        _ = p.transform(np.zeros((5, 5)))
    assert 'Incompatible input dimensions' in str(err.value)

    assert np.allclose(p.transform(source), destination)
    assert np.allclose(p.transform(destination, inverse=True), source)

    s = source.reshape((2, 3, 3))
    d = destination.reshape((2, 3, 3))
    assert np.allclose(p.transform(s), d)

    s = np.ones(2)
    p.transform(s)
    assert np.allclose(p.transform(s), 1.5)

    p = x_polynomial
    d = p.transform(np.arange(2))
    assert np.allclose(d, [1, 3])
    d = p.transform(1)
    assert np.isclose(d, 3)


def test_call(xy_coordinates, xy_polynomial):
    source, destination = xy_coordinates
    p = xy_polynomial
    assert np.allclose(p(source), destination)
    assert np.allclose(p(destination, inverse=True), source)
