# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import scaled_matrix_inverse

import numpy as np


def test_scaled_matrix_inverse():
    m, n = 5, 10
    rstate = np.random.RandomState(1)
    a = rstate.random((m, n))
    a_inv = scaled_matrix_inverse(a)
    assert np.allclose(a @ a_inv, np.eye(m))

    a_inv_scaled = scaled_matrix_inverse(a, n=n, rank=m)
    scale = n / (n - m)
    assert np.allclose(a @ a_inv_scaled, np.eye(m) * scale)
