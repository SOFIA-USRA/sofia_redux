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

from sofia_redux.scan.utilities.class_provider import get_projection_class
# This is a moderately simple class, suitable for testing.
CP = get_projection_class('cylindrical_equal_area')


def test_init():
    p = CP()
    assert p._native_reference == SphericalCoordinates([0, 0])
    assert p._native_pole == SphericalCoordinates([0, 90])
    assert p._celestial_pole.size == 0
    assert not p.user_pole
    assert not p.user_reference
    assert not p.inverted_fits_axes
    assert p.select_solution == 'nearest'


def test_copy():
    p = CP()
    p2 = p.copy()
    assert p2 == p and p2 is not p


def test_reference():
    p = CP()
    reference = SphericalCoordinates([1, 2])
    p.reference = reference
    assert p.reference == reference


def test_native_pole():
    p = CP()
    pole = SphericalCoordinates([0, 80])
    p.native_pole = pole
    assert p.native_pole == pole


def test_celestial_pole():
    p = CP()
    pole = SphericalCoordinates([0, 90])
    p.celestial_pole = pole
    assert p.celestial_pole == pole


def test_native_reference():
    p = CP()
    reference = SphericalCoordinates([1, 2])
    p.native_reference = reference
    assert p.native_reference == reference


def test_eq():
    p = CP()
    p.celestial_pole = SphericalCoordinates([0, 90])
    assert p == p
    assert p is not None
    p2 = p.copy()
    assert p == p2
    p2.reference = SphericalCoordinates([1, 2])
    assert p != p2
    p2 = p.copy()
    p2.user_pole = True
    assert p2 != p
    p2 = p.copy()
    p2.native_reference = SphericalCoordinates([34, 45])
    assert p2 != p
    p2 = p.copy()
    p2.native_pole = SphericalCoordinates([4, 5])
    p2.user_pole = False
    assert p2 != p
    p2 = p.copy()
    p2.celestial_pole = SphericalCoordinates([1, 90])
    assert p2 != p


def test_create_registry():
    class CP2(CP):
        pass

    p = CP2
    p.registry = None
    p.create_registry()
    assert isinstance(p.registry, dict)
    for k in ['SIN', 'TAN', 'ZEA', 'SFL', 'MER', 'CAR', 'AIT', 'STG', 'PCO',
              'BON', 'CYP', 'CEA', 'PAR']:
        assert issubclass(p.registry[k], SphericalProjection)


def test_register():
    class CP2(CP):
        pass

    p = CP2
    p.registry = None
    p.register(None)
    assert isinstance(p.registry, dict) and len(p.registry) == 0
    p.register(CP())
    assert len(p.registry) == 3
    for key in ['CEA', 'cylindrical equal area',
                'cylindricalequalareaprojection']:
        assert p.registry[key] == CP


def test_for_name():
    class CP2(CP):
        pass
    p = CP2

    p.registry = None
    with pytest.raises(ValueError) as err:
        _ = p.for_name('FOO')
    assert "No projection FOO in registry" in str(err.value)
    assert p.for_name('CEA') == CP()


def test_asin():
    ninety = np.pi / 2
    thirty = np.deg2rad(30)
    assert CP.asin(0) == 0
    assert CP.asin(1).value == ninety
    assert CP.asin(2).value == ninety

    x = ((np.arange(7) - 3) / 2) * units.dimensionless_unscaled
    assert np.allclose(CP.asin(x).value,
                       [-ninety, -ninety, -thirty, 0, thirty, ninety, ninety])


def test_acos():
    assert CP.acos(0).value == np.pi / 2
    assert CP.acos(1) == 0
    assert CP.acos(2) == 0

    x = ((np.arange(7) - 3) / 2) * units.dimensionless_unscaled
    a = CP.acos(x).to('degree').value

    assert np.allclose(a, [180, 180, 120, 90, 60, 0, 0])


def test_phi_theta_to_radians():
    rad = units.Unit('radian')
    deg = units.Unit('degree')
    ud = units.dimensionless_unscaled
    p, t = CP.phi_theta_to_radians(1, 2)
    assert p == 1 * rad and t == 2 * rad
    p, t = CP.phi_theta_to_radians(1 * ud, 2 * ud)
    assert p == 1 * rad and t == 2 * rad
    p, t = CP.phi_theta_to_radians(90 * deg, 180 * deg)
    assert np.allclose([p, t], [np.pi / 2, np.pi] * rad)


def test_offset_to_xy_radians():
    rad = units.Unit('radian')
    deg = units.Unit('degree')
    ud = units.dimensionless_unscaled
    x, y = CP.offset_to_xy_radians(Coordinate2D([1, 2]))
    assert x == 1 * rad and y == 2 * rad
    x, y = CP.offset_to_xy_radians(Coordinate2D([1, 2], unit=ud))
    assert x == 1 * rad and y == 2 * rad
    x, y = CP.offset_to_xy_radians(Coordinate2D([90, 180], unit=deg))
    assert np.allclose([x, y], [np.pi / 2, np.pi] * rad)


def test_get_coordinate_instance():
    assert CP().get_coordinate_instance() == SphericalCoordinates()


def test_get_longitude_parameter_prefix():
    p = CP()
    assert p.get_longitude_parameter_prefix() == 'PV1_'
    p.inverted_fits_axes = True
    assert p.get_longitude_parameter_prefix() == 'PV2_'


def test_get_latitude_parameter_prefix():
    p = CP()
    assert p.get_latitude_parameter_prefix() == 'PV2_'
    p.inverted_fits_axes = True
    assert p.get_latitude_parameter_prefix() == 'PV1_'


def test_get_reference():
    p = CP()
    assert p.get_reference() == SphericalCoordinates()


def test_is_right_angle_pole():
    p = CP()
    assert not p.is_right_angle_pole()
    p.celestial_pole = SphericalCoordinates([0, 90])
    assert p.is_right_angle_pole()


def test_project():
    p = CP()
    assert p.celestial_pole.size == 0  # defaults to (0, 0)
    assert p.native_pole == SphericalCoordinates([0, 90])
    c = SphericalCoordinates([45, 30])
    c0 = c.copy()
    projected = SphericalCoordinates()
    o = p.project(c, projected=projected)
    assert projected is o and c == c0
    assert np.allclose(o.coordinates.value, [-50.76847952, 35.08635606])
    p.celestial_pole = p.native_pole.copy()
    o = p.project(c)
    assert np.allclose(o.coordinates.value, [225, 28.64788976])

    # Single pole, multiple coordinates
    c = SphericalCoordinates([[30, 45], [30, 60]])
    o = p.project(c)
    assert np.allclose(o.coordinates.value,
                       [[210, 225], [28.64788976, 49.61960059]])

    # Single coordinate, multiple poles
    p.celestial_pole = SphericalCoordinates([[0, 0, 0], [0, 45, 90]])
    p.native_pole = p.celestial_pole.copy()
    c = c[0]
    o = p.project(c)
    assert np.allclose(o.coordinates.value,
                       [[-40.89339465, -112.2076543, 210.],
                        [42.97183463, 50.64279278, 28.64788976]])


def test_deproject():
    p = CP()

    for celestial_pole in [
            SphericalCoordinates(), SphericalCoordinates([0, 90])]:

        p.celestial_pole = celestial_pole  # SphericalCoordinates([0, 90])

        # Check by projecting and deprojecting
        c = SphericalCoordinates([45, 30])
        c0 = c.copy()
        coordinates = SphericalCoordinates()
        offset = p.project(c)
        assert offset != c0

        cd = p.deproject(offset, coordinates=coordinates)
        assert cd is coordinates and cd == c0

        # Test arrays
        c = SphericalCoordinates([[45, 45], [30, 60]])
        offsets = p.project(c)
        assert offsets != c
        coordinates = p.deproject(offsets)
        assert coordinates is not c and coordinates is not offsets
        assert coordinates == c


def test_set_reference():
    p = CP()
    assert p.celestial_pole.size == 0  # i.e., it's undefined.
    assert not p.user_pole  # no native pole has been set by the user
    assert p.native_reference == SphericalCoordinates([0, 0])
    assert p.native_pole == SphericalCoordinates([0, 90])

    p.set_reference(SphericalCoordinates([30, 60]))
    assert p.native_pole == SphericalCoordinates([0, 90])
    assert p.celestial_pole == SphericalCoordinates([-150, 30])
    assert p.reference == SphericalCoordinates([30, 60])

    p.set_reference(SphericalCoordinates([30, -60]))
    assert p.native_pole == SphericalCoordinates([180, 90])
    assert p.celestial_pole == SphericalCoordinates([30, 30])
    assert p.reference == SphericalCoordinates([30, -60])

    p.set_native_pole(SphericalCoordinates([0, 80]))
    assert p.user_pole
    p.set_reference(SphericalCoordinates([30, 60]))
    assert p.native_pole == SphericalCoordinates([0, 80])
    assert p.celestial_pole == SphericalCoordinates([-150, 30])


def test_calculate_celestial_pole():
    p = CP()
    assert p.native_reference == SphericalCoordinates([0, 0])
    assert p.native_pole == SphericalCoordinates([0, 90])
    assert p.select_solution == 'nearest'
    assert p.celestial_pole.size == 0
    p.calculate_celestial_pole()
    # Check nothing happens without reference
    assert p.celestial_pole.size == 0

    p._reference = SphericalCoordinates([30, 60])
    # Check nothing was inadvertently set
    assert p.celestial_pole.size == 0

    p.calculate_celestial_pole()
    assert p.celestial_pole == SphericalCoordinates([-150, 30])
    p.select_solution = 'northern'
    p.calculate_celestial_pole()
    assert p.celestial_pole == SphericalCoordinates([30, -30])
    p.select_solution = 'southern'
    p.calculate_celestial_pole()
    assert p.celestial_pole == SphericalCoordinates([-150, 30])

    # Check arrays
    p.select_solution = 'nearest'
    p._reference = SphericalCoordinates([[30, 60], [60, 30]])
    p.calculate_celestial_pole()
    assert np.allclose(p.celestial_pole.coordinates.value,
                       [[-150, -120], [30, 60]])


def test_set_native_pole():
    p = CP()
    assert not p.user_pole
    p.set_native_pole(SphericalCoordinates([0, 90]))
    assert p.native_pole == SphericalCoordinates([0, 90])
    assert p.native_reference.is_null()  # (0, 0)
    p.set_reference(SphericalCoordinates([0, 45]))
    assert p.user_pole


def test_set_default_native_pole():
    p = CP()
    p.native_pole = SphericalCoordinates([45, 45])
    assert p.user_pole
    p._reference = SphericalCoordinates([11, 12])
    p.set_default_native_pole()
    assert not p.user_pole
    assert p.native_pole == SphericalCoordinates([0, 45])

    p._reference = SphericalCoordinates([11, -12])
    p.set_default_native_pole()
    assert not p.user_pole
    assert p.native_pole == SphericalCoordinates([180, 45])

    p._reference = SphericalCoordinates([np.zeros(5), [-1, 1, -1, 1, -1]])
    p.set_default_native_pole()
    assert p.native_pole == SphericalCoordinates([180, 45])

    p._reference = SphericalCoordinates([np.zeros(5), [1, -1, 1, -1, 1]])
    p.set_default_native_pole()
    assert p.native_pole == SphericalCoordinates([0, 45])


def test_set_celestial_pole():
    p = CP()
    assert p.celestial_pole.size == 0
    p.set_celestial_pole(SphericalCoordinates([0, 90]))
    assert p.celestial_pole == SphericalCoordinates([0, 90])


def test_set_native_reference():
    p = CP()
    p.set_native_reference(SphericalCoordinates([1, 2]))
    assert p.native_reference == SphericalCoordinates([1, 2])


def test_set_default_pole():
    p = CP()
    p.user_pole = True
    p.set_default_pole()
    assert not p.user_pole
    assert p.native_pole == SphericalCoordinates([0, 90])
    assert p.reference.size == 0

    p.reference = SphericalCoordinates([30, -1])
    assert p.native_pole == SphericalCoordinates([180, 90])
    assert p.reference == SphericalCoordinates([30, -1])


def test_set_native_pole_latitude():
    p = CP()
    p.select_solution = None
    p.set_native_pole_latitude(45 * units.Unit('degree'))
    assert p.native_pole == SphericalCoordinates([0, 45])
    assert p.select_solution == 'nearest'
    p.set_native_pole_latitude(120 * units.Unit('degree'))
    assert p.native_pole == SphericalCoordinates([0, 45])
    assert p.select_solution == 'northern'
    p.set_native_pole_latitude(-120 * units.Unit('degree'))
    assert p.native_pole == SphericalCoordinates([0, 45])
    assert p.select_solution == 'southern'


def test_parse_header():
    h = fits.Header()
    p = CP()
    p.parse_header(h)
    assert not p.inverted_fits_axes
    assert not p.user_pole and not p.user_reference
    assert p.native_pole == SphericalCoordinates([0, 90])
    assert p.native_reference == SphericalCoordinates([0, 0])

    h['CTYPE1'] = 'RA---TAN'
    h['LONPOLE'] = 30.0
    h['LATPOLE'] = 60.0
    h['PV1_1'] = 40.0
    h['PV2_2'] = 50.0
    p.parse_header(h)
    assert not p.inverted_fits_axes
    assert p.user_pole
    assert p.user_reference
    assert p.native_pole == SphericalCoordinates([30, 60])
    assert p.native_reference == SphericalCoordinates([40, 50])

    del h['LONPOLE']
    del h['LATPOLE']
    h['PV2_3'] = 15.0  # pole longitude (reverse axis)
    h['PV1_4'] = 25.0  # pole latitude (reverse axis)
    h['PV2_1'] = 35.0  # reference longitude (reverse axis)
    h['PV1_2'] = 45.0  # reference latitude (reverse axis)

    for ctype in ['DEC--TAN', 'GLAT-TAN']:
        h['CTYPE1'] = ctype
        p.parse_header(h)
        assert p.inverted_fits_axes
        assert p.user_pole and p.user_reference
        assert p.native_pole == SphericalCoordinates([15, 25])
        assert p.native_reference == SphericalCoordinates([35, 45])


def test_edit_header():
    h = fits.Header()
    p = CP()
    p.reference = SphericalCoordinates([1, 2])
    p.native_pole = SphericalCoordinates([0, 90])
    p.user_reference = True
    p.edit_header(h)
    assert h['CTYPE1'] == 'LON--CEA'
    assert h['CTYPE2'] == 'LAT--CEA'
    assert h['LONPOLE'] == 0.0
    assert h['LATPOLE'] == 90.0
    assert h['PV1_1'] == 0.0
    assert h['PV2_2'] == 0.0
    assert h['PV2_1'] == 1.0
