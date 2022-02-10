# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D


def test_init():
    h = HorizontalCoordinates()
    assert h.unit == 'degree' and h.coordinates is None


def test_setup_coordinate_system():
    h = HorizontalCoordinates()
    h.setup_coordinate_system()
    assert h.default_coordinate_system.name == 'Horizontal Coordinates'
    assert h.longitude_axis.label == 'Azimuth'
    assert h.longitude_axis.short_label == 'AZ'
    assert not h.longitude_axis.reverse
    assert h.latitude_axis.label == 'Elevation'
    assert h.latitude_axis.short_label == 'EL'
    assert not h.latitude_axis.reverse
    assert h.x_offset_axis.label == 'Azimuth Offset'
    assert h.x_offset_axis.short_label == 'dAZ'
    assert not h.x_offset_axis.reverse
    assert h.y_offset_axis.label == 'Elevation Offset'
    assert h.y_offset_axis.short_label == 'dEL'
    assert not h.y_offset_axis.reverse


def test_copy():
    h = HorizontalCoordinates([1, 1])
    h2 = h.copy()
    assert h == h2 and h is not h2


def test_fits_longitude_stem():
    h = HorizontalCoordinates()
    assert h.fits_longitude_stem == 'ALON'


def test_fits_latitude_stem():
    h = HorizontalCoordinates()
    assert h.fits_latitude_stem == 'ALAT'


def test_two_letter_code():
    h = HorizontalCoordinates()
    assert h.two_letter_code == 'HO'


def test_az():
    h = HorizontalCoordinates()
    assert h.az is None
    h.az = 1.0 * units.Unit('degree')
    assert h.az == 1.0 * units.Unit('degree')


def test_el():
    h = HorizontalCoordinates()
    assert h.el is None
    h.el = 2 * units.Unit('degree')
    assert h.el == 2 * units.Unit('degree')


def test_za():
    h = HorizontalCoordinates()
    assert h.za is None
    h.el = 30 * units.Unit('degree')
    assert h.za == 60 * units.Unit('degree')
    h.za = 40 * units.Unit('degree')
    assert h.el == 50 * units.Unit('degree')


def test_str():
    h = HorizontalCoordinates()
    assert str(h) == 'Empty coordinates'
    h = HorizontalCoordinates([1, 2])
    assert str(h) == 'Az=1d00m00s El=2d00m00s'
    h = HorizontalCoordinates([[1, 2], [3, 4]])
    assert str(h) == 'Az=1d00m00s->2d00m00s El=3d00m00s->4d00m00s'


def test_getitem():
    h = HorizontalCoordinates(np.arange(10).reshape(2, 5))
    h1 = h[1]
    assert np.allclose(h1.coordinates.value, [1, 6])


def test_convert_horizontal_to_equatorial():
    h = HorizontalCoordinates(np.arange(10).reshape(2, 5))
    e = h.get_instance('equatorial')
    site = h.get_instance('geodetic')
    site.lon = 0.0 * units.Unit('degree')
    site.lat = 0.0 * units.Unit('degree')
    lst = 0 * units.Unit('hourangle')
    h.convert_horizontal_to_equatorial(h, e, site, lst)
    assert np.allclose(
        e.x.value, [0, -9.42786076, -15.86693055, -20.42481137, -23.76989436])
    assert np.allclose(
        e.y.value, [85, 83.91753818, 82.72125856, 81.45942469, 80.15783836])


def test_to_equatorial_offset():
    h = HorizontalCoordinates()
    offset = Coordinate2D([1, 1])
    o0 = offset.copy()
    position_angle = 30 * units.Unit('degree')
    o = h.to_equatorial_offset(offset, position_angle, in_place=True)
    assert o is offset
    assert np.allclose(o.coordinates, [-0.3660254, 1.3660254])
    o2 = h.to_equatorial_offset(o, position_angle, in_place=False)
    assert o is not o2 and o2 == o0


def test_set_az():
    h = HorizontalCoordinates()
    h.set_az(1 * units.Unit('degree'))
    assert h.az == 1 * units.Unit('degree')


def test_set_el():
    h = HorizontalCoordinates()
    h.set_el(2 * units.Unit('degree'))
    assert h.el == 2 * units.Unit('degree')


def test_set_za():
    h = HorizontalCoordinates()
    h.set_za(30 * units.Unit('degree'))
    assert h.za == 30 * units.Unit('degree')
    assert h.el == 60 * units.Unit('degree')


def test_get_parallactic_angle():
    h = HorizontalCoordinates([45, 45])
    site = h.get_instance('geodetic')
    site.set_singular()  # Zero lat/lon

    pa = h.get_parallactic_angle(site)
    assert np.isclose(pa, -125.26438968 * units.Unit('degree'))


def test_to_equatorial():
    h = HorizontalCoordinates([45, 45])
    site = h.get_instance('geodetic')
    site.set_singular()  # Zero lat/lon
    lst = 0 * units.Unit('hourangle')
    e = h.to_equatorial(site, lst)
    assert np.isclose(e.dec, 30 * units.Unit('degree'))
    assert np.isclose(e.ra, 35.26438968 * units.Unit('degree'))


def test_edit_header():
    h = HorizontalCoordinates([30, 45])
    hdr = fits.Header()
    h.edit_header(hdr, 'FOO')
    assert hdr['FOO1'] == 30.0
    assert hdr['FOO2'] == 45.0
    assert hdr['WCSNAME'] == 'Horizontal Coordinates'
    assert hdr.comments['WCSNAME'] == 'coordinate system description'
    h = HorizontalCoordinates(np.arange(10).reshape(2, 5))
    hdr = fits.Header()
    h.edit_header(hdr, 'FOO')
    assert len(hdr) == 0
