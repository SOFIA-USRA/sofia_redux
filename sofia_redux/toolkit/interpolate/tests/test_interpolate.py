# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import Interpolate


@pytest.fixture
def data():
    y, x = np.mgrid[:6, :6]
    z = y + x / 10
    return x[0], y[:, 0], z


def test_init(data):
    x, y, z = data
    i = Interpolate(x, y, z, method='linear', cval=-1, cubic=-0.75,
                    ignorenans=True)
    assert i.method == 'linear'
    assert i.cubic == -0.75
    assert i._product is np.nanprod
    assert np.allclose(i.values[:6, :6], z)
    assert np.allclose(i.values[-1, :], -1)
    assert np.allclose(i.values[:, -1], -1)

    assert np.allclose(i.grid[0], x)
    assert np.allclose(i.grid[1], y)
    i = Interpolate(z, ignorenans=False)
    assert np.allclose(i.grid[0], x)
    assert np.allclose(i.grid[1], y)
    assert i._product is np.prod

    # Check int is cast to float
    i = Interpolate(x, y, z.astype(int), method='linear', cval=-1,
                    cubic=-0.75, ignorenans=True)
    assert np.issubdtype(i.values.dtype, np.inexact)


def test_errors(data):
    x, y, z, = data

    with pytest.raises(ValueError):
        Interpolate(z, method='foo')

    with pytest.raises(ValueError):
        Interpolate(x, y, z, cval='a')

    with pytest.raises(ValueError):
        Interpolate(z, mode='bar')

    with pytest.raises(ValueError):
        Interpolate(x, z)

    with pytest.raises(ValueError):
        Interpolate(x, y[::-1], z)

    with pytest.raises(ValueError):
        Interpolate(x, y[:-1], z)

    with pytest.raises(ValueError):
        Interpolate(x, y[None], z)

    i = Interpolate(z)
    with pytest.raises(ValueError):
        i(1)

    with pytest.raises(ValueError):
        i(1, 1, method='foo')

    with pytest.raises(ValueError):
        i(1, 1, mode='bar')

    i.mode = 'foo'
    xi = np.array([[-1.1, 20.1, 0.1, 4.9, 2.1],
                   [1.6, 1.6, 1.6, 1.6, 1.6]])
    with pytest.raises(ValueError):
        i._find_indices(xi)


def test_find_indices(data):
    x, y, z = data
    i = Interpolate(z, method='linear', mode='constant')
    # xi contains out-of-bounds conditions, edge conditions and
    # in-bounds coordinates
    xi = np.array([[-1.1, 20.1, 0.1, 4.9, 2.1],
                   [1.6, 1.6, 1.6, 1.6, 1.6]])
    indices, d = i._find_indices(xi)
    assert np.allclose(indices[0], [[-1, -1, 0, 4, 2],
                                    [-1, -1, 1, 5, 3]])
    assert np.allclose(d[0], [[1, 0, 0.1, 0.9, 0.1],
                              [0, 1, 0.9, 0.1, 0.9]])
    assert np.allclose(indices[1], [[1], [2]])
    assert np.allclose(d[1], [[0.6], [0.4]])

    i.mode = 'nearest'
    indices, d = i._find_indices(xi)
    assert np.allclose(indices[0], [[0, 5, 0, 4, 2],
                                    [0, 5, 1, 5, 3]])
    assert np.allclose(d[0], [[1, 0, 0.1, 0.9, 0.1],
                              [0, 1, 0.9, 0.1, 0.9]])
    assert np.allclose(indices[1], [[1], [2]])
    assert np.allclose(d[1], [[0.6], [0.4]])

    i.mode = 'reflect'
    i.method = 'cubic'
    indices, d = i._find_indices(xi)
    assert np.allclose(indices[0], [
        [2, 4, 1, 3, 1],
        [1, 5, 0, 4, 2],
        [0, 4, 1, 5, 3],
        [1, 3, 2, 4, 4]])
    assert np.allclose(indices[1], [[0], [1], [2], [3]])
    assert np.allclose(d[0], [
        [2., 1., 1.1, 1.9, 1.1],
        [1., 0., 0.1, 0.9, 0.1],
        [0., 1., 0.9, 0.1, 0.9],
        [1., 2., 1.9, 1.1, 1.9]])
    assert np.allclose(d[1], [[1.6], [0.6], [0.4], [1.4]])

    i.mode = 'mirror'
    indices, d = i._find_indices(xi)
    assert np.allclose(indices[0], [
        [1, 4, 0, 3, 1],
        [0, 5, 0, 4, 2],
        [0, 5, 1, 5, 3],
        [1, 4, 2, 5, 4]])
    assert np.allclose(indices[1], [[0], [1], [2], [3]])
    assert np.allclose(d[0], [
        [2., 1., 1.1, 1.9, 1.1],
        [1., 0., 0.1, 0.9, 0.1],
        [0., 1., 0.9, 0.1, 0.9],
        [1., 2., 1.9, 1.1, 1.9]])
    assert np.allclose(d[1], [[1.6], [0.6], [0.4], [1.4]])

    i.mode = 'wrap'
    indices, d = i._find_indices(xi)
    assert np.allclose(indices[0], [
        [4, 4, 5, 3, 1],
        [5, 5, 0, 4, 2],
        [0, 0, 1, 5, 3],
        [1, 1, 2, 0, 4]])
    assert np.allclose(indices[1], [[0], [1], [2], [3]])
    assert np.allclose(d[0], [
        [2., 1., 1.1, 1.9, 1.1],
        [1., 0., 0.1, 0.9, 0.1],
        [0., 1., 0.9, 0.1, 0.9],
        [1., 2., 1.9, 1.1, 1.9]])
    assert np.allclose(d[1], [[1.6], [0.6], [0.4], [1.4]])


def test_linear(data):
    x, y, z = data
    i = Interpolate(z, method='linear', mode='constant',
                    cval=np.nan, ignorenans=False)
    assert np.allclose(i([2.5, 3.5, 4.5], [2.5, 3.5, 4.5]),
                       [2.75, 3.85, 4.95])

    xi = [-1.1, 20.1, 0.1, 4.9, 2.1]
    yi = [1.6, 1.6, 1.6, 1.6, 1.6]
    r = i(xi, yi)
    assert np.allclose(r, [np.nan, np.nan, 1.61, 2.09, 1.81],
                       equal_nan=True)
    r = i(xi, yi, mode='nearest')
    assert np.allclose(r, [1.6, 2.1, 1.61, 2.09, 1.81])


def test_cubic(data):
    x, y, z = data
    i = Interpolate(z, method='cubic', mode='nearest',
                    cval=np.nan, ignorenans=False)
    xi, yi = [2.5, 3.5, 4.5], [2.5, 3.5, 4.5]
    r = i(xi, yi)
    assert np.allclose(r, [2.75, 3.85, 5.01875])
    r = i(xi, yi, mode='constant')
    assert np.allclose(r, [2.75, 3.85, np.nan], equal_nan=True)
    r = i(xi, yi, mode='reflect')
    assert np.allclose(r, [2.75, 3.85, 5.0875], equal_nan=True)

    i = Interpolate(z, method='cubic', mode='nearest',
                    cval=np.nan, ignorenans=False, cubic=None)
    i.cubic = None
    _ = i(xi, yi)
    assert i.cubic == -0.5

    i.cubic = None
    _ = i(xi, yi, cubic=-0.4)
    assert i.cubic == -0.4


def test_nearest(data):
    x, y, z = data
    i = Interpolate(z, method='nearest')
    assert np.allclose(i([2.5, 3.5], [2.5, 3.5]), [2.2, 3.3])


def test_variance(data):
    x, y, z, = data
    i = Interpolate(z, method='linear', error=2)
    assert np.allclose(i.variance[:-1, :-1], 4)
    assert np.isnan(i.variance[-1]).all()
    assert np.isnan(i.variance[:, -1]).all()
    i = Interpolate(z, method='linear', error=z)
    assert np.allclose(i.values ** 2, i.variance, equal_nan=True)
