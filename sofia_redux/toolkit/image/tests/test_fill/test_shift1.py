# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.image.fill import shift1


def test_shift1():
    test = np.arange(12).reshape(4, 3)
    test1 = shift1(test)
    assert np.allclose(test1, [[2, 0, 1],
                               [5, 3, 4],
                               [8, 6, 7],
                               [11, 9, 10]])
    assert test.dtype == test1.dtype
