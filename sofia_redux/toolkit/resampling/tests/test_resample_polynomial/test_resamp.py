# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import resamp

import numpy as np


def test_resamp():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:11, :11]])
    # z = x + y - 10
    sigma = 0.1
    data = (coordinates[0] - 5.0) + (coordinates[1] - 5.0)
    rand = np.random.RandomState(0)
    noise = rand.normal(loc=0, scale=sigma, size=data.size)
    data += noise
    error = sigma

    fit_55 = resamp(coordinates, data, 5, 5, error=error, order=1)
    assert np.isclose(fit_55, 0, atol=0.1)

    fit_27, counts = resamp(coordinates, data, 2, 7, error=error, order=1,
                            get_counts=True, window=3)
    assert np.isclose(fit_27, -1, atol=0.1)
    assert counts == 27
