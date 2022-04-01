# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import \
    GeodeticCoordinates
from sofia_redux.scan.coordinate_systems.telescope_coordinates import \
    TelescopeCoordinates
from sofia_redux.scan.coordinate_systems.projection.gnomonic_projection \
    import GnomonicProjection
from sofia_redux.scan.coordinate_systems.projector.projector_2d import \
    Projector2D
from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.custom.sofia.frames.frames import SofiaFrames


degree = units.Unit('degree')
um = units.Unit('um')
arcsec = units.Unit('arcsec')


@pytest.fixture
def sofia_frames(populated_hawc_scan):
    return populated_hawc_scan[0].frames


@pytest.fixture
def frames_for_transform():
    frames = SofiaFrames()
    frames.set_frame_size(5)
    frames.telescope_vpa = [0, 30, 45, 60, 90] * degree
    return frames


def test_init():
    frames = SofiaFrames()
    assert frames.utc is None
    assert frames.object_equatorial is None
    assert frames.sofia_location is None
    assert frames.instrument_vpa is None
    assert frames.telescope_vpa is None
    assert frames.chop_vpa is None
    assert frames.pwv is None


def test_default_field_types():
    frames = SofiaFrames()
    fields = frames.default_field_types
    for key in ['instrument_vpa', 'telescope_vpa', 'chop_vpa']:
        value = fields[key]
        assert np.isnan(value) and value.unit == degree
    assert np.isclose(fields['pwv'], np.nan * um, equal_nan=True)
    assert fields['object_equatorial'] == (EquatorialCoordinates, 'degree')
    assert fields['sofia_location'] == (GeodeticCoordinates, 'degree')


def test_project(sofia_frames):
    frames = sofia_frames
    projection = GnomonicProjection()
    projection.set_reference(TelescopeCoordinates([20, 30]))
    projector = Projector2D(projection)
    position = Coordinate2D([0, 0], unit='arcsec')
    c = frames.project(position, projector)
    assert frames.horizontal_offset != c
    assert frames.horizontal_offset.shape == c.shape
    c = frames.project(position, projector)
    c.change_unit('arcsec')
    assert np.allclose(c[1].coordinates, [2.247, 0.056] * arcsec, atol=1e-3)

    projection.set_reference(EquatorialCoordinates([20, 30]))
    projector = AstroProjector(projection)
    c = frames.project(position, projector)
    c.change_unit('arcsec')
    assert np.allclose(c[1].coordinates, [2.247, 0.056] * arcsec, atol=1e-3)


def test_telescope_to_native_equatorial_offset(frames_for_transform):
    frames = frames_for_transform.copy()
    offsets = Coordinate2D([np.arange(5) - 2, np.arange(5) - 2], unit='arcsec')
    o = offsets.copy()
    o2 = frames.telescope_to_native_equatorial_offset(o, in_place=True)
    assert o2 is o
    assert np.allclose(o.coordinates,
                       [[2, 1.3660254, 0, -1.3660254, -2],
                        [2, 0.3660254, 0, 0.3660254, 2]] * arcsec, atol=1e-3)
    e = o.copy()
    o = offsets.copy()
    o2 = frames.telescope_to_native_equatorial_offset(o, in_place=False)
    assert o is not o2 and o2 == e

    o2 = frames.telescope_to_native_equatorial_offset(
        o, indices=4, in_place=False)  # Rotate by 90 degrees
    assert np.allclose(o2.x, -o.x)
    assert np.allclose(o2.y, o.y)


def test_native_equatorial_to_telescope_offset(frames_for_transform):
    frames = frames_for_transform.copy()
    offsets = Coordinate2D([np.arange(5) - 2, np.arange(5) - 2], unit='arcsec')
    o = offsets.copy()
    o2 = frames.native_equatorial_to_telescope_offset(o, in_place=True)
    assert o is o2
    assert np.allclose(o.coordinates,
                       [[2, 0.3660254, 0, 0.3660254, 2],
                        [2, 1.3660254, 0, -1.3660254, -2]] * arcsec, atol=1e-3)
    e = o.copy()
    o = offsets.copy()
    o2 = frames.native_equatorial_to_telescope_offset(o, in_place=False)
    assert o is not o2 and o2 == e

    o2 = frames.native_equatorial_to_telescope_offset(
        o, indices=4, in_place=False)  # Rotate by 90 degrees
    assert np.allclose(o2.x, o.x)
    assert np.allclose(o2.y, -o.y)


def test_telescope_to_equatorial_offset(frames_for_transform):
    frames = frames_for_transform.copy()
    offsets = Coordinate2D([np.arange(5) - 2, np.arange(5) - 2], unit='arcsec')
    e = frames.telescope_to_equatorial_offset(offsets, in_place=False)
    assert np.allclose(e.coordinates,
                       [[-2, -1.3660254, 0, 1.3660254, 2],
                        [2, 0.3660254, 0, 0.3660254, 2]] * arcsec, atol=1e-3)


def test_equatorial_to_telescope_offset(frames_for_transform):
    frames = frames_for_transform.copy()
    offsets = Coordinate2D([np.arange(5) - 2, np.arange(5) - 2], unit='arcsec')
    t = frames.equatorial_to_telescope_offset(offsets, in_place=False)
    assert np.allclose(t.coordinates,
                       [[-2, -1.3660254, 0, 1.3660254, 2],
                        [2, 0.3660254, 0, 0.3660254, 2]] * arcsec, atol=1e-3)
