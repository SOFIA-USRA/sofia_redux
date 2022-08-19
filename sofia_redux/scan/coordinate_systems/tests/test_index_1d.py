# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.index_1d import Index1D


def test_init():
    c = Index1D([1.5, 2.5, 3.5])
    assert np.allclose(c.coordinates, [2, 3, 4])


def test_set_x():
    c = Index1D([1, 2, 3])
    c.set_x(5)
    assert c.x == 5


def test_add_x():
    c = Index1D([1, 2, 3])
    c.add_x(2)
    assert np.allclose(c.x, [3, 4, 5])


def test_subtract_x():
    c = Index1D([1, 2, 3])
    c.subtract_x([0, 2, 4])
    assert np.allclose(c.x, [1, 0, -1])


def test_scale_x():
    c = Index1D([1, 2, 3])
    c.scale_x(2)
    assert np.allclose(c.x, [2, 4, 6])


def test_change_unit():
    c = Index1D([1, 2, 3])
    with pytest.raises(NotImplementedError):
        c.change_unit('arcsec')
    assert 'Cannot give indices unit dimensions'
