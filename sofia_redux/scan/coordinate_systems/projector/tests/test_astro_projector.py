# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.projector.astro_projector import \
    AstroProjector
from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import \
    EquatorialCoordinates
from sofia_redux.scan.coordinate_systems.horizontal_coordinates import \
    HorizontalCoordinates
from sofia_redux.scan.coordinate_systems.focal_plane_coordinates import \
    FocalPlaneCoordinates
from sofia_redux.scan.coordinate_systems.galactic_coordinates import \
    GalacticCoordinates
from sofia_redux.scan.coordinate_systems.projection.plate_carree_projection \
    import PlateCarreeProjection


@pytest.fixture
def andromeda_galactic():
    return GalacticCoordinates([121.174329, -21.573309], unit='degree')


@pytest.fixture
def andromeda_equatorial(andromeda_galactic):
    return andromeda_galactic.to_equatorial()
    # c = ICRS(ra='00h42m44.330s', dec='+40d59m41.73s')
    # return EquatorialCoordinates([c.ra, c.dec])


@pytest.fixture
def projection_galactic(andromeda_galactic):
    ag = andromeda_galactic.copy()
    projection = PlateCarreeProjection()
    projection.reference = ag
    return projection


@pytest.fixture
def projection_equatorial(andromeda_equatorial):
    ae = andromeda_equatorial.copy()
    projection = PlateCarreeProjection()
    projection.reference = ae
    return projection


@pytest.fixture
def projector_celestial(projection_galactic):
    projector = AstroProjector(projection_galactic)
    return projector


@pytest.fixture
def projector_equatorial(projection_equatorial):
    projector = AstroProjector(projection_equatorial)
    return projector


@pytest.fixture
def projector_horizontal():
    projection = PlateCarreeProjection()
    projection.reference = HorizontalCoordinates([1, 2], unit='degree')
    projector = AstroProjector(projection)
    return projector


@pytest.fixture
def projector_fp():
    projection = PlateCarreeProjection()
    projection.reference = FocalPlaneCoordinates([2, 3], unit='degree')
    projector = AstroProjector(projection)
    return projector


def test_init(projector_celestial, projector_equatorial, projector_horizontal):
    assert projector_celestial.equatorial.size == 0
    assert projector_celestial.celestial is not None
    assert projector_equatorial.equatorial.size == 1
    assert projector_equatorial.celestial is None
    assert projector_horizontal.equatorial.size == 0
    assert projector_horizontal.celestial is None


def test_copy(projector_celestial):
    pe = projector_celestial
    pe2 = projector_celestial.copy()
    assert pe is not pe2 and pe == pe2


def test_eq(projector_celestial, projector_equatorial, projector_horizontal):
    pc, pe, _ = projector_celestial, projector_equatorial, projector_horizontal
    assert pc == pc
    assert pc is not None
    assert pc != pe
    projection2 = PlateCarreeProjection()
    projection2.reference = pe.coordinates.copy()
    pe2 = AstroProjector(projection2)
    assert pe2 == pe
    pe2.coordinates.zero()
    assert pe2 != pe


def test_is_horizontal(projector_horizontal, projector_equatorial):
    assert projector_horizontal.is_horizontal()
    assert not projector_equatorial.is_horizontal()


def test_is_focal_plane(projector_fp, projector_equatorial):
    assert projector_fp.is_focal_plane()
    assert not projector_equatorial.is_focal_plane()


def test_set_reference_coordinates(projector_celestial, projector_equatorial):
    assert projector_celestial.equatorial.size == 0
    projector_celestial.set_reference_coordinates()
    assert projector_equatorial.equatorial.size == 1
    assert projector_celestial.equatorial == projector_equatorial.equatorial


def test_project_from_equatorial(projector_celestial, projector_equatorial):
    pc, pe = projector_celestial, projector_equatorial
    pc.set_reference_coordinates()
    pe.set_reference_coordinates()

    offsets = Coordinate2D([1, 1], unit='degree')  # Should be overwritten
    oe = pe.project_from_equatorial(offsets=offsets)
    assert oe is offsets and pe.offset.size == 0
    assert np.allclose(oe.coordinates.value, [0, 0])
    assert pe.celestial is None

    oc = pc.project_from_equatorial(offsets=offsets)
    assert oc is offsets and pc.offset.size == 0
    assert np.allclose(oc.coordinates.value, [0, 0])
    assert pc.celestial.size == 1


def test_deproject(projector_celestial, projector_equatorial):
    pc, pe = projector_celestial, projector_equatorial
    offsets = Coordinate2D([1, 1], unit='degree')

    # Check equatorial coordinates first
    coordinates = EquatorialCoordinates()
    ce = pe.deproject(offsets=offsets, coordinates=coordinates)
    assert ce is coordinates
    assert np.allclose(ce.coordinates.value, [-9.24926435, 42.25856692])
    # Check coordinates were not updated
    assert pe.coordinates != ce

    # Check celestial projection
    coordinates = GalacticCoordinates()
    cc = pc.deproject(offsets=offsets, coordinates=coordinates)
    assert cc is coordinates
    # Check coordinates were not updated
    assert pc.coordinates != cc
    assert np.allclose(cc.coordinates.value, [-120.10638683, -20.56988235])


def test_consistency(projector_celestial, projector_equatorial):
    pc, pe = projector_celestial.copy(), projector_equatorial.copy()
    offsets = Coordinate2D([1, 1], unit='degree')

    pc0 = pc.coordinates.copy()
    pe0 = pe.coordinates.copy()
    pc.deproject(offsets=offsets.copy())
    pe.deproject(offsets=offsets.copy())
    assert pc.coordinates != pc0
    assert pe.coordinates != pe0
    assert pc.offset.size == 0  # should not be set
    assert pe.offset.size == 0

    pc.project()
    pe.project()
    assert pc.offset == offsets  # Check offsets were calculated correctly
    assert pe.offset == offsets
