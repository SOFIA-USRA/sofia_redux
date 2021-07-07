# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    calculate_fitting_weights)

import numpy as np


def test_calculate_fitting_weights():
    errors = np.full(10, 2.0)
    weights = np.full(10, 0.5)
    fit_weights = calculate_fitting_weights(errors, weights,
                                            error_weighting=False)
    assert np.allclose(fit_weights, 0.5)

    fit_weights = calculate_fitting_weights(errors, weights,
                                            error_weighting=True)
    assert np.allclose(fit_weights, 1 / 8)
