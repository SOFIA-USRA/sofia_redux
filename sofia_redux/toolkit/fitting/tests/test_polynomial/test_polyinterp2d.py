# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import polyinterp2d


@pytest.fixture
def data():
    c = np.array([[1, 1, 0.01],
                  [2, 0.1, 0],
                  [0, 0, 0]])
    y, x = np.mgrid[:32, :32]
    z = 1.0 + x + (2 * y) + (0.1 * x * y) + 0.01 * (x ** 2)
    return x, y, z, c


def test_error(data):
    x, y, z, c = data
    assert polyinterp2d(x, y, z, x, y, kx=-1) is None
    assert polyinterp2d(x, y, z, x, y, ky=-1) is None
    assert polyinterp2d(x, y[0], z, x, y) is None


def test_expected(data):
    x, y, z, c = data
    r = polyinterp2d(x, y, z, x, y)
    assert np.allclose(r, z)
