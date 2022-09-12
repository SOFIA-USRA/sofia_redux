# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.index import Index


def test_init():
    c = Index([1, 2])
    assert isinstance(c.coordinates, np.ndarray)
    assert c.coordinates.dtype == int
    assert np.allclose(c.coordinates, [1, 2])


def test_check_coordinate_units():
    c = Index()
    coordinates = np.arange(3) * units.Unit('arcsec')
    with pytest.raises(ValueError) as err:
        c.check_coordinate_units(coordinates)
    assert 'must be dimensionless' in str(err.value)
    x = np.arange(3) * units.dimensionless_unscaled
    coordinates, original = c.check_coordinate_units(x)
    assert (coordinates.dtype in [int, np.int64]
            and np.allclose(coordinates, [0, 1, 2]))
    assert not original
    coordinates, original = c.check_coordinate_units(None)
    assert coordinates is None and original

    x = np.arange(3)
    coordinates, original = c.check_coordinate_units(x)
    assert coordinates is x and original


def test_change_unit():
    c = Index()
    with pytest.raises(NotImplementedError) as err:
        c.change_unit('arcsec')
    assert 'Cannot give indices unit dimensions' in str(err.value)


def test_nan():
    c = Index(np.ones((2, 2)))
    c.nan()
    assert np.allclose(c.coordinates, -1)
    c.fill(1)
    c.nan(1)
    assert np.allclose(c.coordinates, [[1, -1], [1, -1]])
    c = Index()
    c.nan()
    assert c.coordinates is None


def test_is_nan():
    c = Index(np.ones((2, 2)))
    c.nan()
    assert np.allclose(c.is_nan(), True)
    c.coordinates = c.coordinates[:, 1:]  # singular
    assert c.is_nan()
    assert not Index().is_nan()


def test_is_neg1():
    assert Index.is_neg1(np.full((2, 4), -1)).all()
    x = np.asarray([[-1, 0], [-1, -1]])
    assert np.allclose(Index.is_neg1(x), [True, False])


def test_insert_blanks():
    c = Index()
    c.insert_blanks(1)
    assert c.coordinates is None
    c = Index([[1, 2, 3]])
    c.insert_blanks(1)
    assert isinstance(c.coordinates, np.ndarray)
    assert np.allclose(c.coordinates, [[1, -1, 2, 3]])


def test_set_shape():
    c = Index()
    c.set_shape((4, 5))
    assert c.coordinates.dtype in [int, np.int64]
    assert np.allclose(c.coordinates, np.zeros((4, 5)))
