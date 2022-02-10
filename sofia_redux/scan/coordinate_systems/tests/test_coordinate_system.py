# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.scan.coordinate_systems.coordinate_system import (
    CoordinateSystem)
from sofia_redux.scan.coordinate_systems.coordinate_axis import CoordinateAxis


@pytest.fixture
def c2d():
    c = CoordinateSystem(name='2D coordinates', dimensions=2)
    c['x'].label = 'x-axis'
    c['y'].label = 'y-axis'
    return c


def test_init():
    c = CoordinateSystem()
    assert c.name == 'Default Coordinate System'
    assert c.axes is None
    c = CoordinateSystem(name='foo', dimensions=2)
    assert c.name == 'foo'
    assert c.axes[0].label == 'x'
    assert c.axes[1].label == 'y'


def test_len(c2d):
    assert len(CoordinateSystem()) == 0
    assert len(c2d) == 2


def test_getitem(c2d):
    with pytest.raises(KeyError) as err:
        _ = CoordinateSystem()['x']
    assert "No available axes" in str(err.value)
    assert c2d['x'].label == 'x-axis'
    assert c2d['y-axis'].short_label == 'y'
    with pytest.raises(KeyError) as err:
        _ = c2d['z']
    assert 'Axis not found' in str(err.value)


def test_contains(c2d):
    assert 'x-axis' in c2d
    assert 'y' in c2d
    assert 'z' not in c2d


def test_size(c2d):
    assert c2d.size == 2


def test_dimension_name():
    c = CoordinateSystem()
    names = ['x', 'y', 'z', 'u', 'v', 'w', 'xy', 'yy', 'zy']
    for dimension, name in enumerate(names):
        assert c.dimension_name(dimension) == name


def test_add_axis(c2d):
    c = c2d
    z = CoordinateAxis(label='z-axis', short_label='z')
    c.add_axis(z)
    assert c.size == 3
    assert isinstance(c['z'], CoordinateAxis)

    c = CoordinateSystem()
    c.add_axis(z)
    assert c.size == 1
    with pytest.raises(ValueError) as err:
        c.add_axis(z)
    assert 'already has axis z.' in str(err.value)
    z.short_label = None
    with pytest.raises(ValueError) as err:
        c.add_axis(z)
    assert 'already has axis z-axis.' in str(err.value)
