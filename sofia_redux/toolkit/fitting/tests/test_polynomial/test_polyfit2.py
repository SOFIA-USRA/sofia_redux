# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.fitting.polynomial import polyfit2d


def test_errors():
    assert polyfit2d(np.ones(2), np.ones(2), np.ones(1)) is None


def test_expected():
    y, x = np.mgrid[:32, :32]
    z = 1.0 + x + (2 * y) + (0.1 * x * y) + 0.01 * (x ** 2)

    c = polyfit2d(x, y, z, kx=2, ky=2, full=False)
    assert np.allclose(c, [[1, 1, 0.01],
                           [2, 0.1, 0],
                           [0, 0, 0]])

    c = polyfit2d(x, y, z, kx=2, ky=2, full=True)
    assert np.allclose(c, [[1, 1, 0.01],
                           [2, 0.1, 0],
                           [0, 0, 0]])
