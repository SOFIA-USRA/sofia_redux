# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.image.fill import clough_tocher_2dfunc


def test_expected():
    y, x = np.mgrid[:32, :32]
    z = x + y * 1.0
    values = z.ravel()
    yout = y[1:, 1:] - 0.5
    xout = x[1:, 1:] - 0.5
    cout = xout.ravel(), yout.ravel()
    cin = x.ravel(), y.ravel()

    result = clough_tocher_2dfunc(values, cin, cout)
    expected = xout + yout
    assert np.allclose(result, expected.ravel())
