# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D


def test_init():
    p = DefaultProjection2D()
    assert p.reference == Coordinate2D([0, 0])


def test_copy():
    p = DefaultProjection2D()
    p2 = p.copy()
    assert p == p2 and p is not p2


def test_reference():
    p = DefaultProjection2D()
    p.reference = Coordinate2D([1, 2])
    assert p.reference == Coordinate2D([1, 2])


def test_equal():
    p = DefaultProjection2D()
    assert p == p
    assert p is not None
    p2 = p.copy()
    assert p == p2
    p2.reference = Coordinate2D([1, 2])
    assert p != p2


def test_get_reference():
    p = DefaultProjection2D()
    assert p.get_reference() == Coordinate2D([0, 0])


def test_set_reference():
    p = DefaultProjection2D()
    c = Coordinate2D([1, 2])
    p.set_reference(c)
    assert p.reference == c


def test_get_projected():
    p = DefaultProjection2D()
    c = Coordinate2D([1, 2])
    cp = p.get_projected(c)
    assert cp == c and cp is not c


def test_get_deprojected():
    p = DefaultProjection2D()
    c = Coordinate2D([1, 2])
    cdp = p.get_deprojected(c)
    assert cdp == c and cdp is not c
