# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.projection.spherical_projection \
    import SphericalProjection


@pytest.fixture
def dummy_spherical():
    """
    A simply SphericalCoordinate test set.

    Returns
    -------
    SphericalCoordinates
    """
    return SphericalCoordinates(np.arange(10).reshape(2, 5))


@pytest.fixture
def lat45():
    s = SphericalCoordinates()
    s.set_y(np.full(5, 45.0))
    s.set_x(np.arange(5) - 2)
    return s


@pytest.fixture
def projection():
    """
    A zenithal equidistant projection.

    Returns
    -------
    SphericalProjection
    """
    return SphericalProjection.for_name('zenithal equidistant')


def test_init():
    s = SphericalCoordinates()
    assert s.size == 0
    assert s.cos_lat is None and s.sin_lat is None
    assert s.default_coordinate_system.name == 'Spherical Coordinates'
    assert s.default_local_coordinate_system.name == 'Spherical Offsets'

    s = SphericalCoordinates([1, 2])
    assert s.unit == 'degree'
    assert np.allclose(s.coordinates, np.array([1, 2]) * units.Unit('degree'))


def test_copy(dummy_spherical):
    s1 = dummy_spherical
    s2 = s1.copy()
    assert s1 == s2 and s1 is not s2 and s1.coordinates is not s2.coordinates


def test_empty_copy_skip_attributes(dummy_spherical):
    s = dummy_spherical
    skip = s.empty_copy_skip_attributes
    assert 'cos_lat' in skip and 'sin_lat' in skip


def test_coordinate_system(dummy_spherical):
    assert dummy_spherical.coordinate_system.name == 'Spherical Coordinates'


def test_local_coordinate_system(dummy_spherical):
    assert dummy_spherical.local_coordinate_system.name == 'Spherical Offsets'


def test_longitude_axis(dummy_spherical):
    x = dummy_spherical.longitude_axis
    assert (x.label == 'Longitude'
            and x.short_label == 'LON'
            and x.unit == 'deg')


def test_latitude_axis(dummy_spherical):
    y = dummy_spherical.latitude_axis
    assert y.label == 'Latitude' and y.short_label == 'LAT' and y.unit == 'deg'


def test_x_offset_axis(dummy_spherical):
    dx = dummy_spherical.x_offset_axis
    assert dx.label == 'Longitude Offset' and dx.short_label == 'dLON'
    assert dx.unit == 'arcsec'


def test_y_offset_axis(dummy_spherical):
    dy = dummy_spherical.y_offset_axis
    assert dy.label == 'Latitude Offset' and dy.short_label == 'dLAT'
    assert dy.unit == 'arcsec'


def test_two_letter_code(dummy_spherical):
    assert dummy_spherical.two_letter_code == 'SP'


def test_fits_latitude_stem(dummy_spherical):
    assert dummy_spherical.fits_latitude_stem == 'LAT-'


def test_fits_longitude_stem(dummy_spherical):
    assert dummy_spherical.fits_longitude_stem == 'LON-'


def test_reverse_longitude(dummy_spherical):
    assert not dummy_spherical.reverse_longitude


def test_reverse_latitude(dummy_spherical):
    assert not dummy_spherical.reverse_latitude


def test_native_longitude(dummy_spherical):
    assert np.allclose(dummy_spherical.native_longitude.value, np.arange(5))
    s = SphericalCoordinates()
    assert s.native_longitude is None
    s.native_longitude = 1
    assert s.native_longitude == 1 * units.Unit('degree')


def test_native_latitude(dummy_spherical):
    assert np.allclose(dummy_spherical.native_latitude.value, np.arange(5) + 5)
    s = SphericalCoordinates()
    assert s.native_latitude is None
    s.native_latitude = 2
    assert s.native_latitude == 2 * units.Unit('degree')


def test_longitude():
    s = SphericalCoordinates()
    assert s.longitude is None
    s.native_longitude = 1
    assert s.longitude.value == 1
    s.longitude_axis.reverse = True
    assert s.longitude.value == -1
    s.longitude = 1
    assert s.native_longitude.value == -1


def test_latitude():
    s = SphericalCoordinates()
    assert s.latitude is None
    s.native_latitude = 2
    assert s.latitude.value == 2
    s.latitude_axis.reverse = True
    assert s.latitude.value == -2
    s.latitude = 2
    assert s.native_latitude.value == -2


def test_lon():
    s = SphericalCoordinates()
    assert s.lon is None
    s.native_longitude = 3
    assert s.lon.value == 3
    s.longitude_axis.reverse = True
    assert s.lon.value == -3
    s.lon = 3
    assert s.native_longitude.value == -3


def test_lat():
    s = SphericalCoordinates()
    assert s.lat is None
    s.native_latitude = 4
    assert s.lat.value == 4
    s.latitude_axis.reverse = True
    assert s.lat.value == -4
    s.lat = 4
    assert s.native_latitude.value == -4


def test_offset_unit(dummy_spherical):
    u = dummy_spherical.offset_unit
    assert isinstance(u, units.UnitBase) and u == 'arcsec'


def test_eq():
    s = SphericalCoordinates()
    assert s == s
    equatorial = SphericalCoordinates.get_instance('Equatorial')
    assert s != equatorial

    s1 = SphericalCoordinates((1, 1))
    assert s != s1
    assert s1 != s

    s2 = SphericalCoordinates(np.ones((2, 2)))
    assert s1 != s2

    s1 = SphericalCoordinates(np.arange(10).reshape(2, 5))
    s2 = s1.copy()
    s1.subtract_x(360 * units.Unit('degree'))
    assert s1 == s2
    s1.subtract_y(180 * units.Unit('degree'))
    assert s1 == s2

    sx = s1.copy()
    sx.subtract_x(1 * units.Unit('degree'))
    assert sx != s2

    sy = s1.copy()
    sy.subtract_y(1 * units.Unit('degree'))
    assert sy != s2


def test_getitem(dummy_spherical):
    s = dummy_spherical.__getitem__(0)
    assert np.allclose(s.coordinates.value, [0, 5])


def test_str(dummy_spherical):
    s = SphericalCoordinates()
    assert str(s) == 'Empty coordinates (deg)'
    s.unit = None
    assert str(s) == 'Empty coordinates'
    s = SphericalCoordinates([0, 0])
    assert str(s) == 'LON=0d00m00s LAT=0d00m00s'
    assert str(dummy_spherical) == (
        'LON=0d00m00s->4d00m00s LAT=5d00m00s->9d00m00s')


def test_register_and_register_types():
    SphericalCoordinates.register_types()
    s = SphericalCoordinates
    for two_letter in ['SP', 'HO', 'TE', 'FP', 'EQ', 'EC', 'GA', 'SG']:
        assert two_letter in s.ids
        class_type = s.ids[two_letter]
        assert s.id_lookup[class_type] == two_letter
    assert isinstance(s.fits_types, dict) and len(s.fits_types) > 0


def test_get_fits_class():
    class Other(SphericalCoordinates):
        pass

    o = Other
    o.fits_types = None
    with pytest.raises(ValueError) as err:
        o.get_fits_class('foo')
    assert 'Unknown coordinate definition: FOO-' in str(err.value)
    assert o.get_fits_class('LON') == SphericalCoordinates


def test_get_two_letter_class():
    class Other(SphericalCoordinates):
        pass
    o = Other
    o.ids = None
    with pytest.raises(ValueError) as err:
        o.get_two_letter_class('ZZ')
    assert 'Unknown coordinate definition ZZ.' in str(err.value)
    assert o.get_two_letter_class('SP') == SphericalCoordinates


def test_get_class_for():
    class Other(SphericalCoordinates):
        pass
    o = Other
    with pytest.raises(ValueError) as err:
        o.get_class_for('FOO')
    assert 'Unknown coordinate definition FOO.' in str(err.value)
    assert o.get_class_for('spherical_coordinates') == SphericalCoordinates
    assert o.get_class_for('LON') == SphericalCoordinates
    assert o.get_class_for('SP') == SphericalCoordinates


def test_get_two_letter_code_for():
    class Other(SphericalCoordinates):
        pass
    o = Other
    o.id_lookup = None
    assert o.get_two_letter_code_for(SphericalCoordinates) == 'SP'


def test_get_default_system(dummy_spherical):
    s = dummy_spherical
    c, o = s.get_default_system()
    assert c.name == 'Spherical Coordinates' and o.name == 'Spherical Offsets'
    x = c.axes[0]
    assert x.label == 'Longitude' and x.short_label == 'LON'
    dx = o.axes[0]
    assert dx.label == 'Longitude Offset' and dx.short_label == 'dLON'
    y = c.axes[1]
    assert y.label == 'Latitude' and y.short_label == 'LAT'
    dy = o.axes[1]
    assert dy.label == 'Latitude Offset' and dy.short_label == 'dLAT'
    assert x.unit == 'degree' and dx.unit == 'arcsec'
    assert y.unit == 'degree' and dy.unit == 'arcsec'


def test_create_axis():
    x = SphericalCoordinates.create_axis('foo', 'f', unit='radian')
    assert x.label == 'foo' and x.short_label == 'f'
    assert isinstance(x.unit, units.UnitBase) and x.unit == 'radian'


def test_create_offset_axis():
    x = SphericalCoordinates.create_offset_axis('bar', 'b', unit='arcsec')
    assert x.label == 'bar' and x.short_label == 'b'
    assert isinstance(x.unit, units.UnitBase) and x.unit == 'arcsec'


def test_setup_coordinate_system(dummy_spherical):
    s = dummy_spherical.copy()
    s.default_coordinate_system = None
    s.default_local_coordinate_system = None
    s.setup_coordinate_system()
    assert s.default_coordinate_system.name == 'Spherical Coordinates'
    assert s.default_local_coordinate_system.name == 'Spherical Offsets'


def test_set_shape():
    s = SphericalCoordinates()
    s.set_shape((2, 3), empty=True)
    assert s.coordinates.shape == (2, 2, 3)
    assert s.cos_lat.shape == (2, 3)
    assert s.sin_lat.shape == (2, 3)
    s.set_shape((2, 3), empty=False)
    assert s.cos_lat.shape == (2, 3) and np.allclose(s.cos_lat, 1)
    assert s.sin_lat.shape == (2, 3) and np.allclose(s.sin_lat, 0)


def test_set_singular(dummy_spherical):
    s = dummy_spherical.copy()
    s.set_singular()
    assert s.set_singular
    assert s.sin_lat == 0
    assert s.cos_lat == 1


def test_set_y():
    s = SphericalCoordinates()
    s.set_y([0, 90])
    assert np.allclose(s.latitude.value, [0, 90])
    assert np.allclose(s.cos_lat, [1, 0])
    assert np.allclose(s.sin_lat, [0, 1])
    assert not isinstance(s.cos_lat, units.Quantity)
    assert not isinstance(s.sin_lat, units.Quantity)


def test_add_y():
    s = SphericalCoordinates()
    s.y = [0, 90]
    s.add_y(90)
    assert np.allclose(s.y.value, [90, 180])
    assert np.allclose(s.cos_lat, [0, -1])
    assert np.allclose(s.sin_lat, [1, 0])
    assert not isinstance(s.cos_lat, units.Quantity)
    assert not isinstance(s.sin_lat, units.Quantity)


def test_subtract_y():
    s = SphericalCoordinates()
    s.y = [0, 90]
    s.subtract_y(90)
    assert np.allclose(s.y.value, [-90, 0])
    assert np.allclose(s.cos_lat, [0, 1])
    assert np.allclose(s.sin_lat, [-1, 0])
    assert not isinstance(s.cos_lat, units.Quantity)
    assert not isinstance(s.sin_lat, units.Quantity)


def test_zero():
    s = SphericalCoordinates()
    s.zero()
    assert s.sin_lat is None and s.cos_lat is None
    s.y = np.arange(5) + 1
    s.zero()
    assert s.sin_lat.shape == (5,) and np.allclose(s.sin_lat, 0)
    assert s.cos_lat.shape == (5,) and np.allclose(s.cos_lat, 1)
    s.y = np.arange(5) + 1
    s.zero([1, 2])
    assert np.allclose(s.sin_lat == 0, [0, 1, 1, 0, 0])
    assert np.allclose(s.cos_lat == 1, [0, 1, 1, 0, 0])


def test_nan():
    s = SphericalCoordinates()
    s.nan()
    assert s.sin_lat is None and s.cos_lat is None
    s.y = np.arange(5) + 1
    s.nan()
    assert s.sin_lat.shape == (5,) and np.isnan(s.sin_lat).all()
    assert s.cos_lat.shape == (5,) and np.isnan(s.cos_lat).all()
    s.y = np.arange(5) + 1
    s.nan([1, 2])
    assert np.allclose(np.isnan(s.sin_lat), [0, 1, 1, 0, 0])
    assert np.allclose(np.isnan(s.cos_lat), [0, 1, 1, 0, 0])


def test_set():
    s = SphericalCoordinates()
    s.set(np.arange(10).reshape(2, 5))
    assert np.allclose(s.x.value, np.arange(5))
    assert np.allclose(s.y.value, np.arange(5) + 5)


def test_set_native(dummy_spherical):
    s = SphericalCoordinates()
    s.set_native(dummy_spherical)
    assert s == dummy_spherical
    s = SphericalCoordinates()
    s.set_native(np.arange(10).reshape(2, 5))
    assert np.allclose(s.x.value, np.arange(5))
    assert np.allclose(s.y.value, np.arange(5) + 5)


def test_set_native_longitude():
    s = SphericalCoordinates()
    s.set_native_longitude(np.arange(5))
    assert np.allclose(s.x.value, np.arange(5))


def test_set_native_latitude():
    s = SphericalCoordinates()
    s.set_native_latitude(np.arange(5))
    assert np.allclose(s.y.value, np.arange(5))


def test_set_longitude():
    s = SphericalCoordinates()
    s.set_longitude(np.arange(5))
    assert np.allclose(s.x.value, np.arange(5))
    s.longitude_axis.reverse = True
    s.set_longitude(np.arange(5))
    assert np.allclose(s.x.value, -np.arange(5))


def test_set_latitude():
    s = SphericalCoordinates()
    s.set_latitude(np.arange(5))
    assert np.allclose(s.y.value, np.arange(5))
    s.latitude_axis.reverse = True
    s.set_latitude(np.arange(5))
    assert np.allclose(s.y.value, -np.arange(5))


def test_project(projection):
    s = SphericalCoordinates([90, 0])
    proj = projection
    s2 = s.copy()
    s.project(proj, s2)
    assert np.isclose(s.x, -s2.x)
    assert np.isclose(s.y, s2.y)


def test_set_projected(dummy_spherical, projection):
    s = dummy_spherical.copy()
    blank = SphericalCoordinates()
    s2 = s.copy()
    s.project(projection, s2)
    assert s != s2
    blank.set_projected(projection, s2)
    assert blank == s


def test_get_projected(projection):
    s = SphericalCoordinates([90, 0])
    o = s.get_projected(projection)
    assert np.allclose(o.coordinates.value, [-90, 0])


def test_add_native_offset(lat45):
    s = lat45.copy()
    c = Coordinate2D([0.5, 0.5], unit='degree')
    s.add_native_offset(c)
    assert np.allclose(s.y.value, 45.5)
    assert np.allclose(s.x.value, lat45.x.value + 1 / np.sqrt(2))


def test_add_offset(lat45):
    s = lat45.copy()
    c = Coordinate2D([0.5, 0.5], unit='degree')
    s.add_offset(c)
    assert np.allclose(s.y.value, 45.5)
    assert np.allclose(s.x.value, lat45.x.value + 1 / np.sqrt(2))
    s = lat45.copy()
    s.longitude_axis.reverse = True
    s.latitude_axis.reverse = True
    s.add_offset(c)
    assert np.allclose(s.y.value, 44.5)
    assert np.allclose(s.x.value, lat45.x.value - 1 / np.sqrt(2))


def test_subtract_native_offset(lat45):
    s = lat45.copy()
    c = Coordinate2D([0.5, 0.5], unit='degree')
    s.subtract_native_offset(c)
    assert np.allclose(s.y.value, 44.5)
    assert np.allclose(s.x.value, lat45.x.value - 1 / np.sqrt(2))


def test_subtract_offset(lat45):
    s = lat45.copy()
    c = Coordinate2D([0.5, 0.5], unit='degree')
    s.subtract_offset(c)
    assert np.allclose(s.y.value, 44.5)
    assert np.allclose(s.x.value, lat45.x.value - 1 / np.sqrt(2))
    s = lat45.copy()
    s.longitude_axis.reverse = True
    s.latitude_axis.reverse = True
    s.subtract_offset(c)
    assert np.allclose(s.y.value, 45.5)
    assert np.allclose(s.x.value, lat45.x.value + 1 / np.sqrt(2))


def test_get_native_offset_from(lat45):
    s = lat45.copy()
    reference = SphericalCoordinates([0, 45])
    offset = s.get_native_offset_from(reference)
    offset.change_unit('degree')
    assert np.allclose(offset.y.value, 0)
    assert np.allclose(offset.x.value, lat45.x.value * np.sqrt(2) / 2)


def test_get_offset_from(lat45):
    s = lat45.copy()
    reference = SphericalCoordinates([0, 45])
    offset = s.get_offset_from(reference)
    offset.change_unit('degree')
    assert np.allclose(offset.y.value, 0)
    assert np.allclose(offset.x.value, lat45.x.value * np.sqrt(2) / 2)
    expected_x = offset.x
    s.longitude_axis.reverse = True
    s.latitude_axis.reverse = True
    offset = s.get_offset_from(reference)
    assert np.allclose(offset.y.value, 0)
    assert np.allclose(offset.x, -expected_x)


def test_standardize():
    s = SphericalCoordinates([[-361, -360, -359, -1, 0, 180, 1, 359, 360, 361],
                              [-181, -180, -179, -1, 0, 90, 1, 179, 180, 181]])
    s.standardize()
    assert np.allclose(s.x.value, [-1, 0, -359, -1, 0, 180, 1, 359, 0, 1])
    assert np.allclose(s.y.value, [-1, 0, -179, -1, 0, 90, 1, 179, 0, 1])


def test_distance_to(lat45):
    reference = SphericalCoordinates([0, 45])
    distance = lat45.distance_to(reference)
    assert np.allclose(distance.value, abs(np.arange(5) - 2) * np.sqrt(2) / 2,
                       atol=1e-4)

    assert reference.distance_to(reference) == 0 * units.Unit('degree')


def test_edit_header():
    h = fits.Header()
    s = SphericalCoordinates([-1, 0])
    s.edit_header(h, key_stem='FOO', alt='B')
    assert h['FOO1B'] == 359.0
    assert h['FOO2B'] == 0.0
    assert h['WCSAXES'] == 2

    h = fits.Header()
    s = SphericalCoordinates([[0, 1], [0, 1]])
    s.edit_header(h, 'FOO')
    assert len(h) == 0


def test_parse_header():
    s = SphericalCoordinates(unit='degree')
    h = fits.Header()
    h['X1'] = 1.0, 'this is in (arcsec)'
    h['X2'] = 2.0, 'this is in [arcmin]'
    s.parse_header(h, 'X')
    assert np.isclose(s.x.value, 1 / 3600)
    assert np.isclose(s.y.value, 2 / 60)

    s.parse_header(h, 'Y')
    assert s.is_null()

    s.parse_header(h, 'Y', default=SphericalCoordinates([1, 2]))
    assert np.allclose(s.coordinates.value, [1, 2])
    s.parse_header(h, 'Y', default=Coordinate2D([2, 3]))
    assert np.allclose(s.coordinates.value, [2, 3])
    s.parse_header(h, 'Y', default=[3, 4] * units.Unit('degree'))
    assert np.allclose(s.coordinates.value, [3, 4])


def test_invert_y(lat45):
    s = lat45.copy()
    s.invert_y()
    assert np.allclose(s.sin_lat, -np.sqrt(2) / 2)


def test_equal_angles():
    a1 = 1 * units.Unit('degree')
    a2 = 2 * units.Unit('degree')
    assert not SphericalCoordinates.equal_angles(a1, a2)
    assert SphericalCoordinates.equal_angles(
        a1 + (361 * units.Unit('degree')), a2)


def test_transform():
    s0 = SphericalCoordinates([0, 80])
    pole = SphericalCoordinates([0, 55])
    phi0 = 0 * units.Unit('degree')
    s1 = s0.transform(pole, phi0)
    assert s1 == SphericalCoordinates([0, 65])

    s0 = SphericalCoordinates([1, 1])
    neg_pole = SphericalCoordinates([0, -90])
    s1 = s0.transform(neg_pole, phi0)
    assert np.allclose(s1.coordinates.value, -1)
    s1r = s0.transform(neg_pole, phi0, reverse=True)
    assert np.allclose(s1r.coordinates.value, [-1, -1])

    s0 = SphericalCoordinates(np.full((2, 3), 1.0))
    s1 = s0.transform(neg_pole, phi0)
    assert s1.coordinates.shape == (2, 3)
    assert np.allclose(s1.coordinates.value, -1)

    s1 = neg_pole.transform(s0, phi0)
    assert s1.coordinates.shape == (2, 3)
    assert np.allclose(s1.x.value, 180)
    assert np.allclose(s1.y.value, -1)

    s1 = s0.transform(s0, phi0)
    assert s1.coordinates.shape == (2, 3)
    assert np.allclose(s1.x.value, 270)
    assert np.allclose(s1.y.value, 90)

    s0 = SphericalCoordinates(np.ones((2, 3, 4)))
    s1 = s0.transform(neg_pole, phi0)
    assert s1.coordinates.shape == (2, 3, 4)
    assert np.allclose(s1.x.value, -1)
    assert np.allclose(s1.y.value, -1)


def test_inverse_transform():
    s0 = SphericalCoordinates([0, 80])
    pole = SphericalCoordinates([0, 55])
    phi0 = 0 * units.Unit('degree')
    s1 = s0.transform(pole, phi0)
    assert s1 != s0
    s2 = s1.inverse_transform(pole, phi0)
    assert s2 == s0


def test_zero_to_two_pi():
    assert np.isclose(SphericalCoordinates.zero_to_two_pi(
        -1 * units.Unit('degree')).value, 359)


def test_get_indices(dummy_spherical):
    s = dummy_spherical.copy()
    i = np.asarray(1)
    s1 = s.get_indices(i)
    assert s1.singular
    assert np.allclose(s1.coordinates.value, [1, 6])

    assert np.isclose(s1.sin_lat, np.sin(s.y[1]))
    assert np.isclose(s1.cos_lat, np.cos(s.y[1]))
    s = SphericalCoordinates()
    s1 = s.get_indices(0)
    assert s1.coordinates is None and s1.cos_lat is None and s1.sin_lat is None


def test_insert_blanks(dummy_spherical):
    s = dummy_spherical.copy()
    s.insert_blanks([1, 1])
    blanks = [0, 1, 1, 0, 0, 0, 0]
    assert np.allclose(np.isnan(s.coordinates).all(axis=0), blanks)
    assert np.allclose(np.isnan(s.cos_lat), blanks)
    assert np.allclose(np.isnan(s.sin_lat), blanks)


def test_merge(dummy_spherical):
    s1 = dummy_spherical.copy()
    s2 = s1.copy()
    s2.zero()
    s2.merge(s1)
    assert np.allclose(s2.x.value, [0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    assert np.allclose(s2.y.value, [0, 0, 0, 0, 0, 5, 6, 7, 8, 9])
    assert np.allclose(s2.sin_lat, np.sin(s2.y))
    assert np.allclose(s2.cos_lat, np.cos(s2.y))

    s1 = SphericalCoordinates([0, 0])
    s2 = SphericalCoordinates([90, -90])
    s1.merge(s2)
    assert np.allclose(s1.x.value, [0, 90])
    assert np.allclose(s1.y.value, [0, -90])
    assert np.allclose(s1.cos_lat, [1, 0])
    assert np.allclose(s1.sin_lat, [0, -1])

    s1 = SphericalCoordinates([0, 0])
    c2 = Coordinate2D([45, 45], unit='degree')
    s1.merge(c2)
    assert np.allclose(s1.x.value, [0, 45])
    assert np.allclose(s1.y.value, [0, 45])
    assert np.allclose(s1.sin_lat, [0, np.sqrt(2) / 2])
    assert np.allclose(s1.cos_lat, [1, np.sqrt(2) / 2])


def test_paste(dummy_spherical):
    s = dummy_spherical.copy()
    s1 = SphericalCoordinates([90, -90])
    c1 = Coordinate2D([90, -90], unit='degree')
    cos_lat, sin_lat = s.cos_lat.copy(), s.sin_lat.copy()
    cos_lat[1] = 0.0
    sin_lat[1] = -1
    for test in [s1, c1]:
        s = dummy_spherical.copy()
        s.paste(test, 1)
        assert np.allclose(s.x.value, [0, 90, 2, 3, 4])
        assert np.allclose(s.y.value, [5, -90, 7, 8, 9])
        assert np.allclose(s.cos_lat, cos_lat)
        assert np.allclose(s.sin_lat, sin_lat)


def test_shift(dummy_spherical):
    s = dummy_spherical.copy()
    s.shift(2, fill_value=90 * units.Unit('degree'))
    assert np.allclose(s.x.value, [90, 90, 0, 1, 2])
    assert np.allclose(s.y.value, [90, 90, 5, 6, 7])
    assert np.allclose(s.cos_lat, np.cos(s.y))
    assert np.allclose(s.sin_lat, np.sin(s.y))
    s = SphericalCoordinates([0, 0])
    s1 = s.copy()
    s.shift(99)
    assert s == s1
