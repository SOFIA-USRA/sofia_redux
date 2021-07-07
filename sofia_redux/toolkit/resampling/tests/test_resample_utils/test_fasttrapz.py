# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import fasttrapz

import numpy as np


def test_fasttrapz():

    # test function is y = 2x^2 + 3x + 1
    def f(xx):
        yy = (2.0 * xx ** 2) + (3.0 * xx) + 1.0
        return yy

    x = np.linspace(-2, 2, 100)
    y = f(x)

    # integration = 2/3.x^3 + 3/2.x^2 + x + C, lim(-2,2)
    expected = (2.0 * (x ** 3) / 3.0) + (3.0 * (x ** 2) / 2.0) + x
    expected = expected[-1] - expected[0]

    result = fasttrapz(y, x)
    assert np.allclose(result, expected, atol=0.01)
