# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate import Coordinate
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_systems_numba_functions \
    import check_null


@pytest.fixture
def c2d():
    """
    Return a test 2-D coordinate.

    Returns
    -------
    Coordinate
    """
    c = Coordinate(np.arange(10).reshape((2, 5)) * units.Unit('degree'))
    return c


def test_init():
    c = Coordinate()
    assert c.coordinates is None
    assert c.unit is None

    c = Coordinate(unit='degree')
    assert c.unit == units.Unit('degree')

    coordinates = np.arange(10)
    c = Coordinate(coordinates, unit='degree')
    assert c.coordinates.unit == 'degree'
    assert np.allclose(c.coordinates.value, np.arange(10))
    assert c.ndim == 1

    coordinates = np.arange(10).reshape((2, 5)) * units.Unit('m')
    c = Coordinate(coordinates)
    c2 = Coordinate(c)
    assert c2.shape == (5,) and c2.ndim == 2 and c2.unit == 'm'


def test_empty_copy(c2d):
    c = c2d
    c2 = c.empty_copy()
    assert c2.unit == 'degree' and c2.coordinates is None


def test_copy(c2d):
    c = c2d
    c2 = c.copy()
    assert c is not c2
    assert np.allclose(c.coordinates, c2.coordinates)
    assert c.unit == c2.unit


def test_empty_copy_skip_attributes():
    c = Coordinate()
    assert 'coordinates' in c.empty_copy_skip_attributes


def test_ndim(c2d):
    assert c2d.ndim == 2
    assert Coordinate().ndim == 0
    assert Coordinate(np.arange(5)).ndim == 1


def test_shape(c2d):
    c = c2d.copy()
    assert c.shape == (5,)
    c.shape = (5, 2)
    assert c.coordinates.shape == (2, 5, 2)
    c = Coordinate()
    assert c.shape == ()


def test_size(c2d):
    assert c2d.size == 5
    assert Coordinate().size == 0
    assert Coordinate([0]).size == 1


def test_singular(c2d):
    assert not c2d.singular
    assert Coordinate().singular
    assert Coordinate([0]).singular


def test_length():
    d = Coordinate().length
    assert not isinstance(d, units.Quantity) and np.isnan(d)
    d = Coordinate(unit='m').length
    assert d.unit == 'm' and np.isnan(d)
    x1 = np.asarray([-2, 1, 2]) * units.Unit('m')
    d = Coordinate(x1).length
    assert np.allclose(d, np.abs(x1))
    x2 = np.stack([x1, x1])
    d = Coordinate(x2).length
    assert np.allclose(d, np.hypot(*x2))
    x3 = np.stack([x1, x1, x1])
    d = Coordinate(x3).length
    assert np.allclose(d, np.sqrt(np.sum(x3 ** 2, axis=0)))


def test_eq(c2d):
    c = c2d
    other = Coordinate2D(c.coordinates)
    assert c == c
    assert c != other
    assert c != Coordinate(np.arange(4))
    assert c != Coordinate(c.coordinates.value)
    c2 = c.copy()
    assert c == c2
    c2.coordinates[0, 0] *= np.nan
    assert c != c2

    c = c2d.copy()
    c0 = Coordinate()
    assert c0 != c
    assert c != c0
    c1 = Coordinate()
    assert c0 == c1


def test_len(c2d):
    assert len(c2d) == 5


def test_getitem(c2d):
    c = c2d
    assert np.allclose(c[3].coordinates, c.coordinates[:, 3][..., None])
    assert np.allclose(c[1:3].coordinates, c.coordinates[:, 1:3])


def test_setitem(c2d):
    c = c2d.copy()
    c2 = c[1:3].copy()
    c2.coordinates *= np.nan
    c[1:3] = c2
    assert c[1:3] == c2
    assert c != c2d
    assert c[3:] == c2d[3:]


def test_get_indices(c2d):
    c = Coordinate(unit='degree')
    assert c.get_indices(0).coordinates is None
    c = Coordinate(np.asarray([1]))
    with pytest.raises(KeyError) as err:
        c.get_indices(1)
    assert "Cannot retrieve indices" in str(err.value)

    c = c2d.copy()
    c1 = c.get_indices(np.asarray(1))
    assert c1.singular and np.allclose(c1.coordinates.value, [[1], [6]])
    c2 = c.get_indices((1,))
    assert c1 == c2
    c1 = c.get_indices(slice(0, 2))
    assert np.allclose(c1.coordinates.value, [[0, 1], [5, 6]])


def test_set_shape(c2d):
    c = c2d.copy()
    c.set_shape(6, empty=True)
    assert c.coordinates.shape == (2, 6)
    c.set_shape((2, 2), empty=False)
    assert c.coordinates.shape == (2, 2, 2)
    assert c.coordinates.unit == 'degree'

    c = Coordinate()
    c.set_shape(5)
    assert c.ndim == 1
    assert c.size == 5
    assert c.coordinates.shape == (1, 5)

    c = Coordinate()
    c.set_shape((2, 5))
    assert c.ndim == 2
    assert c.size == 5
    assert c.coordinates.shape == (2, 5)


def test_set_singular(c2d):
    c = Coordinate()
    c.set_singular(empty=False)
    assert c.shape == ()
    assert c.coordinates.shape == (1,)
    c = c2d.copy()
    c.set_singular(empty=True)
    assert c.shape == ()
    assert c.coordinates.shape == (2, 1)
    assert c.coordinates.unit == 'degree'


def test_set_shape_from_coordinates():
    c = Coordinate()
    c.set_shape_from_coordinates(1, empty=True, single_dimension=False)
    assert c.ndim == 1
    assert c.shape == ()
    assert c.coordinates.shape == (1,)
    assert c.unit is None

    c = Coordinate()
    c.set_shape_from_coordinates(1 * units.Unit('degree'), empty=False,
                                 single_dimension=True)
    assert c.ndim == 1
    assert c.shape == ()
    assert c.coordinates.shape == (1,)
    assert c.unit == 'degree'

    c = Coordinate()
    c.set_shape_from_coordinates(np.arange(5), single_dimension=True)
    assert c.ndim == 1 and c.shape == (5,) and c.coordinates.shape == (1, 5)

    c = Coordinate()
    c.set_shape_from_coordinates(np.arange(5), single_dimension=False)
    assert c.ndim == 1 and c.shape == (5,) and c.coordinates.shape == (1, 5)
    assert c.unit is None

    # Now update previous dimensionality with a new shape
    c.set_shape_from_coordinates(np.arange(5), single_dimension=True)
    assert c.ndim == 1 and c.shape == (5,) and c.coordinates.shape == (1, 5)

    # Coordinates should be supplied to a dimensionally initialized
    # Coordinate as an (ndim, n) array.  If not, the coordinate will be set to
    # singular.
    c.set_shape_from_coordinates(np.arange(5), single_dimension=False)
    assert c.shape == () and c.ndim == 1 and c.coordinates.shape == (1,)

    c.set_shape_from_coordinates(np.atleast_2d(np.arange(5)) * units.Unit('m'),
                                 single_dimension=False)
    assert c.shape == (5,) and c.ndim == 1 and c.coordinates.shape == (1, 5)
    assert c.unit == 'm'


def test_check_coordinate_units(c2d):
    c = c2d.copy()
    assert c.unit == 'degree'
    assert c.ndim == 2
    assert c.check_coordinate_units(None) == (None, True)

    coords, original = c.check_coordinate_units(c)
    assert coords is c and original

    quantity = np.arange(5) * units.Unit('arcsec')
    coords, original = c.check_coordinate_units(quantity)
    assert coords.unit == 'degree' and np.allclose(coords, quantity)
    assert not original

    weird_coords = [np.ones((3, 4)) * units.Unit('arcsec'),
                    np.ones((3, 4)) * units.Unit('arcmin')]
    coords, original = c.check_coordinate_units(weird_coords)
    assert not original
    assert coords.shape == (2, 3, 4)
    assert np.allclose(coords[0].value, 1 / 3600)
    assert np.allclose(coords[1].value, 1 / 60)

    weird_coords = [1 * units.Unit('arcsec'), 1 * units.Unit('arcmin')]
    coords, original = c.check_coordinate_units(weird_coords)
    assert not original
    assert coords[0].value == 1 / 3600
    assert coords[1].value == 1 / 60

    c = Coordinate()
    coords, original = c.check_coordinate_units(weird_coords)
    assert c.unit == 'arcsec'
    assert not original
    assert coords[0].value == 1 and coords[1].value == 60

    c = Coordinate(unit='arcsec')
    coords, original = c.check_coordinate_units(['1', '2'])
    assert coords.unit == 'arcsec' and not original
    assert coords[0].value == 1 and coords[1].value == 2

    coords, original = c.check_coordinate_units(
        np.arange(3) * units.dimensionless_unscaled)
    assert not original and coords.unit == 'arcsec'
    assert np.allclose(coords.value, np.arange(3))

    cu = Coordinate(np.arange(3), unit=units.dimensionless_unscaled)
    coords, original = c.check_coordinate_units(cu)
    assert not original
    assert np.allclose(coords.coordinates.value, [0, 1, 2])

    c2 = Coordinate([1, 2])
    coords, original = c.check_coordinate_units(c2)
    assert not original and coords.unit == 'arcsec'
    assert np.allclose(coords.coordinates.value, [1, 2])

    c2 = Coordinate([1, 2], unit='arcmin')
    coords, original = c.check_coordinate_units(c2)
    assert not original and coords.unit == 'arcsec'
    assert np.allclose(coords.coordinates.value, [60, 120])


def test_change_unit(c2d):
    c = c2d.copy()
    assert c.unit == 'degree'
    c.change_unit('arcmin')
    assert c.unit == 'arcmin'
    expected = np.arange(10).reshape(2, 5) * 60
    assert np.allclose(c.coordinates.value, expected)
    c.change_unit('arcmin')
    assert np.allclose(c.coordinates.value, expected)

    c = Coordinate()
    c.change_unit('arcsec')
    assert c.coordinates is None and c.unit == 'arcsec'

    c = Coordinate(np.arange(5))
    c.change_unit('arcmin')
    assert np.allclose(c.coordinates.value, np.arange(5))
    assert c.coordinates.unit == 'arcmin'


def test_broadcast_to(c2d):
    c = c2d.copy()
    s1 = c.shape
    c.broadcast_to((2, 2))
    assert c.shape == s1  # no change since not singular

    c = Coordinate(np.full((2, 1), 1.0))
    assert c.singular and c.ndim == 2
    c.broadcast_to((np.empty((3, 4))))
    assert c.shape == (3, 4)
    assert c.coordinates.shape == (2, 3, 4) and np.allclose(c.coordinates, 1)

    c = Coordinate(np.full((2, 1), 1.0))
    c.broadcast_to(())
    assert c.shape == () and c.singular and c.coordinates.shape == (2, 1)

    c.broadcast_to(None)
    assert c.singular and c.shape == ()


def test_convert_factor():
    c = Coordinate(np.arange(5))
    assert c.convert_factor(1.0) == 1.0
    assert c.convert_factor(1.0 * units.dimensionless_unscaled) == 1.0

    v = c.convert_factor(2 * units.Unit('degree'))
    assert c.unit == 'degree'
    assert v == 2
    assert c.coordinates.unit == 'degree'

    v = c.convert_factor(1 * units.Unit('arcsec'))
    assert v == 1 / 3600


def test_nan(c2d):
    c = Coordinate()
    c.nan()
    assert c.coordinates is None
    c = c2d.copy()
    c.nan()
    assert np.all(np.isnan(c.coordinates))

    c = c2d.copy()
    c.nan(np.array([1, 2]))
    assert np.all(np.isnan(c.coordinates[:, 1:3]))
    assert np.allclose(c[4].coordinates.ravel().value, [4, 9])

    c = Coordinate(np.arange(4).astype(float))
    c.nan(0)
    assert np.isnan(c.coordinates[0])
    assert np.allclose(c.coordinates[1:], [1, 2, 3])


def test_zero(c2d):
    c = Coordinate()
    c.zero()
    assert c.coordinates is None
    c = c2d.copy()
    c.zero()
    assert np.allclose(c.coordinates, 0)

    c = c2d.copy()
    c.zero(np.array([1, 2]))
    assert np.allclose(c.coordinates[:, 1:3], 0)
    assert np.allclose(c[4].coordinates.ravel().value, [4, 9])

    c = Coordinate(np.arange(4).astype(float))
    c.zero(0)
    assert c.coordinates[0] == 0
    assert np.allclose(c.coordinates[1:], [1, 2, 3])


def test_apply_coordinate_mask_function():
    c = Coordinate()
    x = np.zeros((2, 3, 4))
    x[1, 1, 1] = 1.0
    mask = c.apply_coordinate_mask_function(x, check_null)
    m0 = mask.copy()
    assert np.allclose(mask,
                       [[True, True, True, True],
                        [True, False, True, True],
                        [True, True, True, True]])
    mask = c.apply_coordinate_mask_function(x * units.Unit('s'), check_null)
    assert np.allclose(mask, m0)


def test_is_null():
    c = Coordinate()
    assert not c.is_null()
    c = Coordinate(np.full((2, 1), 0.0))
    assert c.is_null()
    c = Coordinate(np.full((2, 1), 1.0))
    assert not c.is_null()

    x1 = np.arange(3).astype(float)
    c = Coordinate(x1)
    assert np.allclose(c.is_null(), [True, False, False])

    x2 = np.stack([x1, x1])
    c = Coordinate(x2)
    assert np.allclose(c.is_null(), [True, False, False])

    x3 = np.stack([x1, x1, x1])
    c = Coordinate(x3)
    assert np.allclose(c.is_null(), [True, False, False])

    xs = np.zeros((2, 3, 4))
    xs[0, 1, 1] = 1.0
    c = Coordinate(xs)
    assert np.allclose(c.is_null(),
                       [[True, True, True, True],
                        [True, False, True, True],
                        [True, True, True, True]])


def test_is_nan():
    c = Coordinate()
    assert not c.is_nan()
    c = Coordinate(np.full((2, 1), 0.0))
    assert not c.is_nan()
    c = Coordinate(np.full((2, 1), np.nan))
    assert c.is_nan()

    x1 = np.arange(3).astype(float)
    x1[0] = np.nan
    c = Coordinate(x1)
    assert np.allclose(c.is_nan(), [True, False, False])

    x2 = np.stack([x1, x1])
    c = Coordinate(x2)
    assert np.allclose(c.is_nan(), [True, False, False])

    x3 = np.stack([x1, x1, x1])
    c = Coordinate(x3)
    assert np.allclose(c.is_nan(), [True, False, False])

    xs = np.zeros((2, 3, 4))
    xs[0, 1, 1] = np.nan
    c = Coordinate(xs)
    assert np.allclose(c.is_nan(),
                       [[False, False, False, False],
                        [False, True, False, False],
                        [False, False, False, False]])


def test_is_finite():
    c = Coordinate()
    assert not c.is_finite()
    c = Coordinate(np.full((2, 1), 0.0))
    assert c.is_finite()
    c = Coordinate(np.full((2, 1), np.inf))
    assert not c.is_finite()

    x1 = np.arange(3).astype(float)
    x1[0] = np.nan
    c = Coordinate(x1)
    assert np.allclose(c.is_finite(), [False, True, True])

    x2 = np.stack([x1, x1])
    c = Coordinate(x2)
    assert np.allclose(c.is_finite(), [False, True, True])

    x3 = np.stack([x1, x1, x1])
    c = Coordinate(x3)
    assert np.allclose(c.is_finite(), [False, True, True])

    xs = np.zeros((2, 3, 4))
    xs[0, 1, 1] = np.inf
    c = Coordinate(xs)
    assert np.allclose(c.is_finite(),
                       [[True] * 4,
                        [True, False, True, True],
                        [True] * 4])


def test_is_infinite():
    c = Coordinate()
    assert not c.is_infinite()
    c = Coordinate(np.full((2, 1), 0.0))
    assert not c.is_infinite()
    c = Coordinate(np.full((2, 1), np.inf))
    assert c.is_infinite()

    x1 = np.arange(3).astype(float)
    x1[0] = np.inf
    c = Coordinate(x1)
    assert np.allclose(c.is_infinite(), [True, False, False])

    x2 = np.stack([x1, x1])
    c = Coordinate(x2)
    assert np.allclose(c.is_infinite(), [True, False, False])

    x3 = np.stack([x1, x1, x1])
    c = Coordinate(x3)
    assert np.allclose(c.is_infinite(), [True, False, False])

    xs = np.zeros((2, 3, 4))
    xs[0, 1, 1] = np.inf
    c = Coordinate(xs)
    assert np.allclose(c.is_infinite(),
                       [[False] * 4,
                        [False, True, False, False],
                        [False] * 4])


def test_convert_from(c2d):
    c = c2d.copy()
    c0 = Coordinate()
    c0.convert_from(c)
    assert np.allclose(c0.coordinates, c.coordinates)
    assert c0.coordinates is not c.coordinates


def test_convert_to(c2d):
    c = c2d.copy()
    c0 = Coordinate()
    c.convert_to(c0)
    assert np.allclose(c0.coordinates, c.coordinates)
    assert c0.coordinates is not c.coordinates


def test_correct_factor_dimensions():
    c = Coordinate(np.ones((2, 3, 4)))
    assert c.correct_factor_dimensions(1, c) == 1
    assert c.correct_factor_dimensions(np.asarray(1), c) == 1
    f = np.ones((2, 1))
    assert np.allclose(c.correct_factor_dimensions(f, c), f)
    assert np.allclose(c.correct_factor_dimensions(np.ones(2), c), f)
    assert np.allclose(c.correct_factor_dimensions(f, c.coordinates),
                       f[..., None])


def test_get_class():
    c = Coordinate.get_class()
    assert c == Coordinate

    for check in [
            'coordinate', 'Coordinate', 'coordinate_2d', '2d', 'Coordinate2D',
            'celestial_coordinates', 'celestial', 'CelestialCoordinates',
            'ecliptic_coordinates', 'ecliptic', 'EclipticCoordinates',
            'equatorial_coordinates', 'equatorial', 'EquatorialCoordinates',
            'focal_plane_coordinates', 'focal_plane', 'FocalPlaneCoordinates',
            'galactic_coordinates', 'galactic', 'GalacticCoordinates',
            'geocentric_coordinates', 'geocentric', 'GeocentricCoordinates',
            'geodetic_coordinates', 'geodetic', 'GeodeticCoordinates',
            'horizontal_coordinates', 'horizontal', 'HorizontalCoordinates',
            'index_2d', 'Index2D', 'offset_2d', 'Offset2D',
            'precessing_coordinates', 'precessing', 'PrecessingCoordinates',
            'super_galactic_coordinates', 'super_galactic',
            'SuperGalacticCoordinates', 'telescope_coordinates', 'telescope',
            'TelescopeCoordinates']:  # I wrote a lot of systems :<
        x = Coordinate.get_class(check)
        assert issubclass(x, Coordinate)

    with pytest.raises(ValueError) as err:
        _ = Coordinate.get_class('foo')
    assert "Could not find" in str(err.value)

    with pytest.raises(ValueError) as err:
        _ = Coordinate.get_class('cartesian_system')
    assert "Retrieved class" in str(err.value)


def test_get_instance():
    c = Coordinate.get_instance('2d')
    assert isinstance(c, Coordinate2D)


def test_insert_blanks(c2d):
    c = Coordinate()
    c.insert_blanks(1)
    assert c.coordinates is None

    c = Coordinate(np.arange(5).astype(float))
    c.insert_blanks([0, 0, 0])
    assert np.isnan(c.coordinates[:3]).all() and np.allclose(
        c.coordinates[3:], np.arange(5))

    c = c2d.copy()
    c.insert_blanks([0, 0, 0])
    assert np.isnan(c.coordinates[:, :3]).all()
    assert c[3:] == c2d


def test_merge(c2d):
    c = Coordinate()
    c.merge(c)
    assert c.coordinates is None
    other = c2d.copy()
    c.merge(other)
    assert c == c2d

    c = Coordinate(np.arange(5))
    with pytest.raises(ValueError) as err:
        c.merge(other)
    assert "do not match" in str(err.value)

    c = Coordinate(np.arange(5))
    c.merge(c)
    assert c.shape == (10,) and c.coordinates.shape == (1, 10) and c.ndim == 1
    assert np.allclose(c.coordinates.ravel(), np.arange(10) % 5)

    c = Coordinate2D([3, 3])
    c2 = Coordinate2D([4, 4])
    c.merge(c2)
    assert c.shape == (2,)
    assert np.allclose(c.coordinates, [[3, 4], [3, 4]])


def test_paste(c2d):
    c = Coordinate()
    indices = np.arange(2)
    with pytest.raises(ValueError) as err:
        c.paste(c, indices=indices)
    assert "Cannot paste onto singular" in str(err.value)
    c = Coordinate(np.arange(5))
    with pytest.raises(ValueError) as err:
        c.paste(Coordinate(), indices=indices)
    assert "Cannot paste empty coordinates" in str(err.value)
    c = c2d.copy()
    c1 = Coordinate(np.arange(5))
    with pytest.raises(ValueError) as err:
        c.paste(c1, indices=indices)
    assert "Coordinate dimensions" in str(err.value)

    c = Coordinate(np.arange(5) + 3)
    cp = Coordinate(np.full(2, 1))
    c.paste(cp, indices=indices)
    assert np.allclose(c.coordinates, [1, 1, 5, 6, 7])
    c = c2d.copy()
    cp = c[3:].copy()
    cp.coordinates *= 2
    c.paste(cp, indices=indices)
    assert np.allclose(c.coordinates.value,
                       [[6, 8, 2, 3, 4],
                        [16, 18, 7, 8, 9]])


def test_shift(c2d):
    c = Coordinate()
    c.shift(1)
    assert c.coordinates is None

    c = Coordinate(np.arange(5), unit='m')
    cs = c.copy()
    cs.shift(2)
    assert np.isnan(cs.coordinates[:2]).all()
    assert np.allclose(cs.coordinates[2:].value, np.arange(3))
    assert cs.coordinates.unit == 'm'

    cs = c.copy()
    cs.shift(-2, fill_value=-99)
    assert np.allclose(cs.coordinates.value, [2, 3, 4, -99, -99])

    cs = c.copy()
    cs.shift(0)
    assert cs == c

    c = c2d.copy()
    c.shift(2, fill_value=-1)
    assert np.allclose(c.coordinates.value, [[-1, -1, 0, 1, 2],
                                             [-1, -1, 5, 6, 7]])


def test_mean(c2d):
    c = Coordinate()
    m = c.mean()
    assert m.coordinates is None
    c = Coordinate(np.full((2, 1), 1.0))
    assert c.singular and c.mean() == c

    c = Coordinate(np.arange(5))
    m = c.mean()
    assert np.allclose(m.coordinates, [2]) and m.singular

    c = c2d.copy()
    m = c.mean()
    assert np.allclose(m.coordinates.value, [[2], [7]])


def test_copy_coordinates(c2d):
    c = c2d.copy()
    c0 = Coordinate()
    c0.copy_coordinates(c)
    assert c0 == c


def test_set(c2d):
    c = Coordinate()
    c2 = c2d.copy()
    c.set(c2.coordinates, copy=False)
    assert c.coordinates is c2.coordinates

    c.set(c2.coordinates, copy=True)
    assert c.coordinates is not c2.coordinates
    assert np.allclose(c.coordinates, c2.coordinates)

    c = Coordinate(np.full((2, 3), 1.0))
    c.set(np.arange(2))
    assert np.allclose(c.coordinates, [[0], [1]])
