# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_system import \
    CoordinateSystem
from sofia_redux.scan.coordinate_systems.grid.grid import Grid

system2d = CoordinateSystem(dimensions=2)


class TestingGrid(Grid):
    """An un-abstracted grid for testing"""

    def __init__(self):
        super().__init__()
        self._reference = Coordinate2D([0, 0])
        self._resolution = 1.0
        self.copy_test = np.arange(10)
        self.set_coordinate_system(system2d)

    @property
    def referenced_attributes(self):
        referenced = super().referenced_attributes
        referenced.add('_reference')
        return referenced

    def get_reference(self):
        return self._reference

    def set_reference(self, value):
        self._reference = value

    def get_reference_index(self):
        return self._reference

    def set_reference_index(self, value):
        self._reference = value

    def get_resolution(self):
        return self._resolution

    def set_resolution(self, value):
        self._resolution = value

    def get_dimensions(self):
        return 2


def test_init():
    g = TestingGrid()
    assert g.variant == 0
    assert g.coordinate_system == system2d


def test_referenced_attributes():
    g = TestingGrid()
    assert g.referenced_attributes == {'_reference'}


def test_copy():
    g = TestingGrid()
    g2 = g.copy()
    assert g.reference is g2.reference
    assert g.coordinate_system is not g2.coordinate_system
    assert np.allclose(g.copy_test, g2.copy_test)
    assert g.copy_test is not g2.copy_test


def test_coordinate_system():
    g = TestingGrid()
    assert g.coordinate_system.name == 'Default Coordinate System'
    g.coordinate_system = CoordinateSystem(name='FOO', dimensions=2)
    assert g.coordinate_system.name == 'FOO'


def test_ndim():
    g = TestingGrid()
    assert g.ndim == 2


def test_resolution():
    g = TestingGrid()
    g.resolution = 2.0
    assert g.resolution == 2.0


def test_reference():
    g = TestingGrid()
    g.reference = Coordinate2D([1, 1])
    assert g.reference == Coordinate2D([1, 1])


def test_fits_id():
    assert TestingGrid().fits_id == ''


def test_get_grid_class():
    c = TestingGrid.get_grid_class('spherical_grid')
    assert c.__name__ == 'SphericalGrid'


def test_get_grid_instance():
    c = TestingGrid.get_grid_instance('spherical_grid')
    assert isinstance(c, Grid)
    assert c.coordinate_system.name == 'Spherical Coordinates'


def test_set_coordinate_system():
    g = TestingGrid()
    with pytest.raises(ValueError) as err:
        g.set_coordinate_system(CoordinateSystem(dimensions=3))
    assert 'does not equal the grid dimensions' in str(err.value)
    g.set_coordinate_system(CoordinateSystem(name='FOO', dimensions=2))
    assert g.coordinate_system.name == 'FOO'


def test_get_fits_id():
    g = TestingGrid()
    assert g.get_fits_id() == ''
    g.variant = 1
    assert g.get_fits_id() == 'B'
    g.variant = 10
    assert g.get_fits_id() == 'K'
