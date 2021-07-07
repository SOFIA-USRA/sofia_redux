# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import fit_phi_value


def test_fit_phi_value():
    x, y = np.random.random((2, 1000))
    x_dot_y = np.dot(x, y)
    assert np.isclose(x_dot_y, fit_phi_value(x, y))


def test_poison_values():
    x, y = np.random.random((2, 1000))
    x[0] = np.nan
    assert np.isnan(fit_phi_value(x, y))
    x[0] = np.inf
    assert fit_phi_value(x, y) == np.inf
    x[0] = -np.inf
    assert fit_phi_value(x, y) == -np.inf
