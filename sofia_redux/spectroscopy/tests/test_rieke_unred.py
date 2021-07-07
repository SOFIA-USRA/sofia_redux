# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.rieke_unred import rieke_unred


def test_reike_failure():
    assert rieke_unred(3650, 1, 1, model='model_does_not_exist') is None
    assert rieke_unred([3650, 4400], [1, 2, 3], 1) is None


def test_reike_unred():
    assert np.isclose(rieke_unred(3650, 1, 2, r_v=1), 129.419, atol=1e-3)
    assert np.allclose(rieke_unred(3650, [1, 2, 3], 2, r_v=1),
                       [129.42, 258.839, 388.259], atol=1e-3)
    assert np.allclose(rieke_unred([3650, 4400, 5500], 1, 2, r_v=1),
                       [129.42, 39.811, 6.31], atol=1e-3)
    assert np.allclose(
        rieke_unred([3650, 4400, 5500], [1, 2, 3], 2, r_v=1),
        [[129.42, 79.621, 18.929]], atol=1e-3)
