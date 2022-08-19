# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.index_3d import Index3D


degree = units.Unit('degree')


def test_init():
    c = Index3D([1.5, 2.5, 3.5])
    assert np.allclose(c.coordinates, [2, 3, 4])


def test_set_z():
    c = Index3D([1, 2, 3])
    c.set_z(4)
    assert c.z == 4


def test_rotate_offsets():
    c = Index3D([1, 1, 1])
    angle = 90 * degree
    c.rotate_offsets(c, angle)
    assert c.x == -1 and c.y == 1 and c.z == 1
    offsets = np.ones(3, dtype=int)
    c.rotate_offsets(offsets, angle)
    assert np.allclose(offsets, [-1, 1, 1])


def test_add_z():
    c = Index3D([1, 1, 1])
    c.add_z(2)
    assert np.allclose(c.coordinates, [1, 1, 3])


def test_subtract_z():
    c = Index3D([1, 1, 1])
    c.subtract_z(2)
    assert np.allclose(c.coordinates, [1, 1, -1])


def test_scale_z():
    c = Index3D([2, 2, 2])
    c.scale_z(3)
    assert np.allclose(c.coordinates, [2, 2, 6])


def test_change_unit():
    c = Index3D([1, 1, 1])
    with pytest.raises(NotImplementedError) as err:
        c.change_unit('arcsec')
    assert 'Cannot give indices unit dimensions' in str(err.value)
