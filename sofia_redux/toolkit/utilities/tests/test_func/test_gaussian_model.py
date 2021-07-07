# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.func import gaussian_model


def test_standard():
    x = np.arange(11, dtype=float)
    x0 = 5  # center x
    y0 = 1.0  # baseline offset
    amplitude = 10.0
    fwhm = 2.0

    result = gaussian_model(x, x0, amplitude, fwhm, y0)
    assert np.allclose(result[:5], result[-1:-6:-1])  # symmetrical
    assert np.allclose(result[:6],
                       [1.0000003, 1.00015259, 1.01953125, 1.625, 6, 11])
