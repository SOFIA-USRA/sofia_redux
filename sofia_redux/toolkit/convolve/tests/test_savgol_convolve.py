# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.convolve.kernel import SavgolConvolve


@pytest.fixture
def nanhole():
    y, x = np.mgrid[:50, :50]
    x, y = x.astype(float), y.astype(float)
    z = x * y
    z[25:30, 25:30] = np.nan
    return x, y, z, 2


def test_interpolate(nanhole):
    args = nanhole
    x, y, z, window = args
    z[2, 2] = 1000
    z[48, 49] = np.nan
    noise = np.random.normal(0, 0.01, z.shape)
    z += noise
    SavgolConvolve(x, y, z, 10, robust=0)
