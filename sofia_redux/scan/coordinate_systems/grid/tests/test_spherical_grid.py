# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.grid.spherical_grid import \
    SphericalGrid


@pytest.fixture
def equatorial():
    return SphericalGrid(reference=EquatorialCoordinates([45, 45]))


def test_init():
    g = SphericalGrid()
    assert isinstance(g.reference, SphericalCoordinates)
    assert g.reference.size == 0
    assert g.projection.__class__.__name__ == 'GnomonicProjection'
    with pytest.raises(ValueError) as err:
        _ = SphericalGrid(reference=Coordinate2D([0, 0]))
    assert "Reference must be" in str(err.value)

    s = SphericalGrid(reference=SphericalCoordinates([1, 1]))
    assert s.reference == SphericalCoordinates([1, 1])


def test_set_defaults():
    g = SphericalGrid()
    g._coordinate_system = None
    g.set_defaults()
    assert g.x_axis.label == 'Longitude'
    assert g.y_axis.label == 'Latitude'


def test_fits_x_unit():
    assert SphericalGrid().fits_x_unit == 'degree'


def test_fits_y_unit():
    assert SphericalGrid().fits_y_unit == 'degree'


def test_get_default_unit():
    assert SphericalGrid.get_default_unit() == 'degree'


def test_get_coordinate_instance_for():
    assert isinstance(SphericalGrid.get_coordinate_instance_for('RA'),
                      EquatorialCoordinates)
    assert isinstance(SphericalGrid.get_coordinate_instance_for('LON'),
                      SphericalCoordinates)


def test_set_reference():
    g = SphericalGrid()
    g.set_reference(EquatorialCoordinates([1, 1]))
    assert g.x_axis.label == 'Right Ascension'
    assert g.y_axis.label == 'Declination'


def test_is_reverse_x():
    g = SphericalGrid()
    assert not g.is_reverse_x()
    g.reference.longitude_axis.reverse = True
    assert g.is_reverse_x()


def test_is_reverse_y():
    g = SphericalGrid()
    assert not g.is_reverse_y()
    g.reference.latitude_axis.reverse = True
    assert g.is_reverse_y()


def test_parse_projection():
    g = SphericalGrid()
    h = fits.Header()
    h['CTYPE1'] = 'FOO'
    with pytest.raises(ValueError) as err:
        g.parse_projection(h)
    assert 'Cannot extract projection from CTYPE=FOO' in str(err.value)

    h['CTYPE1'] = 'RA---TAN'
    g.parse_projection(h)
    assert g.projection.__class__.__name__ == 'GnomonicProjection'
    h['CTYPE1'] = 'RA---AIT'
    g.parse_projection(h)
    assert g.projection.__class__.__name__ == 'HammerAitoffProjection'

    h['CTYPE1'] = 'RA---FOO'
    with pytest.raises(ValueError) as err:
        g.parse_projection(h)
    assert "Unknown projection: FOO" in str(err.value)
