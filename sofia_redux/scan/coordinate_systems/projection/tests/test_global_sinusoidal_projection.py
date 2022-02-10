# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.spherical_coordinates import \
    SphericalCoordinates
from sofia_redux.scan.utilities.class_provider import get_projection_class
GSP = get_projection_class('global_sinusoidal')


def test_init():
    p = GSP()
    assert p.reference.size == 0


def test_get_fits_id():
    assert GSP.get_fits_id() == 'GLS'


def test_get_full_name():
    assert GSP.get_full_name() == 'Global Sinusoidal'


def test_project():
    p = GSP()
    p.reference = SphericalCoordinates([30, 10])
    o = p.project(SphericalCoordinates([31, 60]))
    assert np.allclose(o.coordinates.value, [0.5, 50])
    o1 = o.copy()
    o2 = p.project(SphericalCoordinates([31, 60]), projected=o1)
    assert o2 is o1 and o2 == o


def test_deproject():
    p = GSP()
    p.reference = SphericalCoordinates([30, 59])
    c = p.deproject(Coordinate2D([1, 1], unit='degree'))
    assert np.allclose(c.coordinates.value, [32, 60])
    c1 = c.copy()
    c2 = p.deproject(Coordinate2D([1, 1], unit='degree'), coordinates=c1)
    assert c2 is c1 and c2 == c


def test_get_phi_theta():
    p = GSP()
    with pytest.raises(NotImplementedError) as err:
        p.get_phi_theta(None)
    assert "Not implemented" in str(err.value)


def test_get_offsets():
    p = GSP()
    with pytest.raises(NotImplementedError) as err:
        p.get_offsets(None, None)
    assert "Not implemented" in str(err.value)
