# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.projector.projector_2d import \
    Projector2D
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D


@pytest.fixture
def dummy_projection():
    projection = DefaultProjection2D()
    projection.reference = Coordinate2D([5, 5])
    return projection


@pytest.fixture
def dummy_projector(dummy_projection):
    projector = Projector2D(dummy_projection)
    return projector


def test_init():
    projection = DefaultProjection2D()
    projection.reference = Coordinate2D([5, 5])
    projector = Projector2D(projection)
    assert projector.coordinates == projection.reference
    assert projector.coordinates is not projection.reference
    assert projector.offset.size == 0
    assert projector.projection is projection


def test_copy(dummy_projector):
    projector = dummy_projector
    p2 = projector.copy()
    assert p2 is not projector
    assert p2.coordinates == projector.coordinates


def test_set_reference_coordinates(dummy_projector):
    projector = dummy_projector
    projector.coordinates.zero()
    projector.set_reference_coordinates()
    assert projector.coordinates == Coordinate2D([5, 5])


def test_project(dummy_projector):
    projector = dummy_projector
    offsets = projector.project()
    assert offsets is projector.offset

    o2 = Coordinate2D()
    offsets = projector.project(offsets=o2)
    assert offsets is o2 and o2 == Coordinate2D([5, 5])

    c = Coordinate2D([1, 2])
    offsets = projector.project(coordinates=c)
    assert offsets == c and offsets is not c
    assert offsets is projector.offset


def test_deproject(dummy_projector):
    projector = dummy_projector
    o = Coordinate2D([1, 2])
    projector.offset = o.copy()

    coordinates = projector.deproject()
    assert coordinates is projector.coordinates

    c2 = Coordinate2D()
    c3 = projector.deproject(coordinates=c2)
    assert c3 is c2 and c2 == o

    offsets = Coordinate2D([3, 4])
    coordinates = projector.deproject(offsets=offsets)
    assert coordinates == offsets and offsets is not coordinates
    assert coordinates is projector.coordinates
