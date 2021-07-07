# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import solve_amat_beta


def test_solve_amat_beta():
    x = (np.arange(12, dtype=np.float64) + 1).reshape((3, 4))
    y = np.linspace(1, 2, 4)
    weights = np.diag(np.linspace(2, 4, 4))

    a, b = solve_amat_beta(x, y, np.diag(weights))

    amat = x @ weights @ x.T
    beta = x @ weights @ y

    assert np.allclose(a, amat)
    assert np.allclose(b, beta)
