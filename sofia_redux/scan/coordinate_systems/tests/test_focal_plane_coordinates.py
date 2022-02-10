# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.scan.coordinate_systems.focal_plane_coordinates import \
    FocalPlaneCoordinates


def test_init():
    f = FocalPlaneCoordinates()
    assert f.unit == 'degree' and f.coordinates is None


def test_copy():
    f = FocalPlaneCoordinates([1, 2])
    f2 = f.copy()
    assert f == f2 and f is not f2


def test_setup_coordinate_system():
    f = FocalPlaneCoordinates()
    f.setup_coordinate_system()
    assert f.default_coordinate_system.name == 'Focal Plane Coordinates'
    assert f.longitude_axis.label == 'Focal-plane X'
    assert f.longitude_axis.short_label == 'X'
    assert not f.longitude_axis.reverse
    assert f.latitude_axis.label == 'Focal-plane Y'
    assert f.latitude_axis.short_label == 'Y'
    assert not f.latitude_axis.reverse
    assert f.x_offset_axis.label == 'Focal-plane dX'
    assert f.x_offset_axis.short_label == 'dX'
    assert not f.x_offset_axis.reverse
    assert f.y_offset_axis.label == 'Focal-plane dY'
    assert f.y_offset_axis.short_label == 'dY'
    assert not f.y_offset_axis.reverse


def test_fits_longitude_stem():
    f = FocalPlaneCoordinates()
    assert f.fits_longitude_stem == 'FLON'


def test_fits_latitude_stem():
    f = FocalPlaneCoordinates()
    assert f.fits_latitude_stem == 'FLAT'


def test_two_letter_code():
    f = FocalPlaneCoordinates()
    assert f.two_letter_code == 'FP'


def test_getitem():
    f = FocalPlaneCoordinates(np.arange(10).reshape(2, 5))
    assert np.allclose(f[1].coordinates.value, [1, 6])


def test_edit_header():
    h = fits.Header()
    f = FocalPlaneCoordinates([1, 2])
    f.edit_header(h, 'FOO')
    assert h['FOO1'] == 1
    assert h['FOO2'] == 2
    assert h['WCSNAME'] == 'Focal Plane Coordinates'
    assert h.comments['WCSNAME'] == 'coordinate system description.'
