# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.coordinate_systems.grid.grid_1d import Grid1D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d1 import FlatGrid2D1
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.coordinate_systems.projection.gnomonic_projection \
    import GnomonicProjection


def test_init():
    grid = FlatGrid2D1()
    assert isinstance(grid.z, Grid1D)


def test_copy():
    g = FlatGrid2D1()
    g2 = g.copy()
    assert g == g2 and g is not g2


def test_set_defaults():
    g = FlatGrid2D1()
    g.set_defaults()
    assert g.ndim == 2
    assert isinstance(g.projection, DefaultProjection2D)


def test_get_coordinate_instance_for():
    assert isinstance(FlatGrid2D1.get_coordinate_instance_for('foo'),
                      Coordinate2D1)


def test_set_projection():
    g = FlatGrid2D1()
    with pytest.raises(ValueError) as err:
        g.set_projection(GnomonicProjection())
    assert 'Generic projections are not allowed' in str(err.value)
    projection = DefaultProjection2D()
    g.set_projection(projection)
    assert g.projection is projection


def test_parse_projection():
    h = fits.Header()
    g = FlatGrid2D1()
    g.parse_projection(h)  # does nothing

