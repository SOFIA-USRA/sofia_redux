# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.index_2d import Index2D


@pytest.fixture
def i2d():
    return Index2D(np.arange(10).reshape(2, 5).astype(float))


def test_init():
    c = Index2D(np.arange(10).reshape(2, 5).astype(float))
    assert c.coordinates.dtype in [int, np.int64]


def test_set_x():
    c = Index2D()
    i = np.arange(6).reshape(2, 3)
    c.set_x(i)
    assert c.shape == (2, 3)
    assert np.allclose(c.x, i)
    assert np.allclose(c.y, 0)
    c = Index2D()
    c.set_x(1)
    assert c.singular and c.x == 1 and c.y == 0


def test_set_y():
    c = Index2D()
    i = np.arange(6).reshape(2, 3)
    c.set_y(i)
    assert c.shape == (2, 3)
    assert np.allclose(c.y, i)
    assert np.allclose(c.x, 0)
    c = Index2D()
    c.set_y(1)
    assert c.singular and c.x == 0 and c.y == 1


def test_check_coordinate_units():
    c = Index2D()
    assert c.unit is None
    with pytest.raises(ValueError) as err:
        _ = c.check_coordinate_units(1 * units.Unit('s'))
    assert 'must be dimensionless' in str(err.value)

    x, original = c.check_coordinate_units(None)
    assert x is None and original

    x, original = c.check_coordinate_units(1)
    assert x == 1 and original

    t = np.arange(3) * units.dimensionless_unscaled / 2
    x, original = c.check_coordinate_units(t)
    assert np.allclose(x, [0, 1, 1]) and not original

    t = np.arange(3) * units.Unit('s') / units.Unit('ms')
    x, original = c.check_coordinate_units(t)
    assert np.allclose(x, [0, 1000, 2000]) and not original


def test_change_unit():
    c = Index2D()
    with pytest.raises(NotImplementedError) as err:
        c.change_unit('s')
    assert "Cannot give indices unit dimensions" in str(err.value)


def test_nan(i2d):
    c = Index2D()
    c.nan()
    assert c.coordinates is None
    c = i2d.copy()
    c.nan(np.arange(2))
    assert np.allclose(c[:2].coordinates, -1)
    assert not np.allclose(c[2:].coordinates, -1)
    c.nan()
    assert np.allclose(c.coordinates, -1)


def test_is_nan(i2d):
    c = Index2D()
    assert not c.is_nan()
    c = Index2D([0, 1])
    assert not c.is_nan()
    c = Index2D([-1, -1])
    assert c.is_nan()
    c = i2d.copy()
    c.nan(np.arange(2))
    assert np.allclose(c.is_nan(), [True, True, False, False, False])


def test_is_neg1():
    c = Index2D()
    x = np.zeros((3, 4))
    x[:, 1] = -1
    assert np.allclose(c.is_neg1(x), [False, True, False, False])


def test_insert_blanks(i2d):
    c = Index2D()
    c.insert_blanks(0)
    assert c.coordinates is None
    c = Index2D([0, 1])
    c.insert_blanks(0)
    assert np.allclose(c.coordinates, [0, 1])
    c = i2d.copy()
    c.insert_blanks(np.ones(2, dtype=int))
    assert np.allclose(c.coordinates,
                       [[0, -1, -1, 1, 2, 3, 4],
                        [5, -1, -1, 6, 7, 8, 9]])


def test_rotate_offsets(i2d):
    c = i2d.copy()
    offsets = c.coordinates.copy()
    c.rotate_offsets(offsets, 45 * units.Unit('degree'))
    assert np.allclose(offsets,
                       [[-4, -4, -4, -4, -4],
                        [4, 5, 6, 8, 9]], atol=1)

    offsets = c.coordinates.copy()
    c.rotate_offsets(offsets, 90 * units.Unit('degree'))
    assert np.allclose(offsets,
                       [[-5, -6, -7, -8, -9],
                        [0, 1, 2, 3, 4]])

    offsets = np.ones(2, dtype=int)
    c.rotate_offsets(offsets, -90 * units.Unit('degree'))
    assert np.allclose(offsets, [1, -1])

    c = Index2D([1, 1])
    c.rotate_offsets(c, -90 * units.Unit('degree'))
    assert np.allclose(c.coordinates, [1, -1])

    c = i2d.copy()
    c.rotate_offsets(c, 45 * units.Unit('degree'))
    assert np.allclose(c.coordinates,
                       [[-4, -4, -4, -4, -4],
                        [4, 5, 6, 8, 9]])


def test_add_x(i2d):
    c = i2d.copy()
    c.add_x(1)
    assert np.allclose(c.coordinates, [[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
    c.add_x(np.full(5, -1.5))
    assert np.allclose(c.coordinates, [[-1, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    c = Index2D([1, 1])
    c.add_x(1.5)
    assert np.allclose(c.coordinates, [3, 1])


def test_subtract_x(i2d):
    c = i2d.copy()
    c.subtract_x(1)
    assert np.allclose(c.coordinates, [[-1, 0, 1, 2, 3], [5, 6, 7, 8, 9]])
    c.subtract_x(np.full(5, -1.5))
    assert np.allclose(c.coordinates, [[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]])
    c = Index2D([1, 1])
    c.subtract_x(1.5)
    assert np.allclose(c.coordinates, [-1, 1])


def test_add_y(i2d):
    c = i2d.copy()
    c.add_y(1)
    assert np.allclose(c.coordinates, [[0, 1, 2, 3, 4], [6, 7, 8, 9, 10]])
    c.add_y(np.full(5, -1.5))
    assert np.allclose(c.coordinates, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    c = Index2D([1, 1])
    c.add_y(1.5)
    assert np.allclose(c.coordinates, [1, 3])


def test_subtract_y(i2d):
    c = i2d.copy()
    c.subtract_y(1)
    assert np.allclose(c.coordinates, [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
    c.subtract_y(np.full(5, -1.5))
    assert np.allclose(c.coordinates, [[0, 1, 2, 3, 4], [6, 7, 8, 9, 10]])
    c = Index2D([1, 1])
    c.subtract_y(1.5)
    assert np.allclose(c.coordinates, [1, -1])


def test_scale_x(i2d):
    c = i2d.copy()
    c.scale_x(1.5)
    assert np.allclose(c.coordinates, [[0, 2, 3, 5, 6], [5, 6, 7, 8, 9]])


def test_scale_y(i2d):
    c = i2d.copy()
    c.scale_y(1.5)
    assert np.allclose(c.coordinates, [[0, 1, 2, 3, 4], [8, 9, 11, 12, 14]])
