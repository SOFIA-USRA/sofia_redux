# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import fit_phi_variance


def test_fit_phi_variance():
    # This tests the algorithm performs as expected.  Note that it is roughly
    # 1000 times faster than the standard numpy equivalent.
    phi = np.random.random(10)
    cov = np.random.random((10, 10))
    pcpt = phi @ cov @ phi.T

    v = fit_phi_variance(phi, cov)
    assert np.isclose(pcpt, v)
