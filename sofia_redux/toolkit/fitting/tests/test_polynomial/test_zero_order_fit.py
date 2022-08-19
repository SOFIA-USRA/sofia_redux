# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.fitting.polynomial import zero_order_fit


def test_meanfit():
    rand = np.random.RandomState(42)
    mean, mvar = zero_order_fit(rand.normal(10, 1, int(1e6)))
    assert np.allclose(mean, 10, rtol=0.01)
    assert np.allclose(mvar, 1e-6, rtol=0.01)
