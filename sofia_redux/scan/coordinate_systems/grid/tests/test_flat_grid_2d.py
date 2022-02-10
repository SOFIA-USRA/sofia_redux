# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.grid.flat_grid_2d import FlatGrid2D
from sofia_redux.scan.coordinate_systems.projection.default_projection_2d \
    import DefaultProjection2D


def test_init():
    g = FlatGrid2D()
    assert np.allclose(g.m, np.eye(2))
    assert np.allclose(g.i, g.m)
    assert g.reference_index == Coordinate2D([0, 0])
    assert isinstance(g.projection, DefaultProjection2D)


def test_copy():
    g = FlatGrid2D()
    g2 = g.copy()
    assert g == g2 and g is not g2


def test_set_defaults():
    g = FlatGrid2D()
    g._coordinate_system = None
    g._projection = None
    g.set_defaults()
    assert isinstance(g.projection, DefaultProjection2D)
    assert g.x_axis.label == 'x'
    assert g.y_axis.label == 'y'


def test_get_coordinate_instance_for():
    g = FlatGrid2D()
    for ctype in ['equatorial', 'horizontal', 'spherical']:
        assert isinstance(g.get_coordinate_instance_for(ctype), Coordinate2D)


def test_set_projection():
    g = FlatGrid2D()
    p = g.projection.copy()
    p.reference = Coordinate2D([1, 1])
    assert p != g.projection
    g.set_projection(p)
    assert g.projection == p

    with pytest.raises(ValueError) as err:
        g.set_projection(None)
    assert "Generic projections are not allowed" in str(err.value)


def test_parse_projection():
    g = FlatGrid2D()
    h = fits.Header()
    h['CDELT1'] = 2.0
    h['CDELT2'] = 3.0
    g0 = g.copy()
    g.parse_projection(h)
    assert g == g0  # Nothing happens


def test_parse_header():
    g = FlatGrid2D()
    h = fits.Header()
    h['CTYPE1'] = 'RA'
    h['CTYPE2'] = 'DEC'
    g.parse_header(h)
    assert g.x_axis.short_label == 'RA'
    assert g.y_axis.short_label == 'DEC'


def test_edit_header():
    h = fits.Header()
    g = FlatGrid2D()
    g.edit_header(h)
    assert h['CTYPE1'] == 'x'
    assert h['CTYPE2'] == 'y'
    g.x_axis.short_label = 'RA'
    g.y_axis.short_label = 'DEC'
    g.edit_header(h)
    assert h['CTYPE1'] == 'RA'
    assert h['CTYPE2'] == 'DEC'
