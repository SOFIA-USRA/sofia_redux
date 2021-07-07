# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.resampling.resample_utils import fit_residual


def test_fit_residual():

    # Create data set for the equation y = 2x + 1
    data = (2.0 * np.arange(1000)) + 1.0
    coefficients = np.array([1.0, 2.0])
    x = np.arange(1000, dtype=np.float64)
    xp = np.stack((x ** 0, x ** 1))

    residual = fit_residual(data, xp, coefficients)
    assert residual.size == 1000
    assert np.allclose(residual, 0)

    noise = np.random.random(1000)
    residual = fit_residual(data + noise, xp, coefficients)
    assert np.allclose(residual, noise)
