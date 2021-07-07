# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.utilities.func import slicer


@pytest.fixture
def zyx():
    return np.mgrid[:2, :3, :4]


def test_single_index(zyx):
    z, y, x = zyx
    s = z.shape

    for a in [x, y, z]:
        for dim in range(3):
            for i in range(s[dim]):
                if dim == 0:
                    assert np.allclose(slicer(a, dim, i), a[i])
                elif dim == 1:
                    assert np.allclose(slicer(a, dim, i), a[:, i])
                elif dim == 2:
                    assert np.allclose(slicer(a, dim, i), a[:, :, i])

    inds = slicer(x, 1, 0, ind=True)
    assert inds[0] == slice(None, None, None)
    assert inds[1] == 0
    assert inds[2] == slice(None, None, None)


def test_multi_index(zyx):
    z, y, x = zyx

    yslice = slicer(y, 1, [0, 0])
    assert yslice.shape == (3,)
    assert np.allclose(yslice, [0, 1, 2])
    yinds = slicer(y, 1, [0, 0], ind=True)
    assert yinds[0] == 0
    assert yinds[1] == slice(None, None, None)
    assert yinds[2] == 0
