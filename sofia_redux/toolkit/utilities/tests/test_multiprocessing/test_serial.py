# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import _serial

import numpy as np


def adder(args, i):
    x, y = args
    return x[i] + y[i]


def test_serial():
    n = 50
    xy = np.arange(n)
    xy = np.vstack((xy, xy + 100))
    expected = np.sum(xy, axis=0)

    result = _serial(adder, xy, None, range(n))
    assert np.allclose(result, expected)

    skip = np.full(n, False)
    skip[10] = True
    result = _serial(adder, xy, None, range(n), skip=skip)
    assert np.allclose(result, expected[~skip])
