# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import update_mask

import numpy as np


def test_update_mask_zero_values():
    mask = np.full(100, True)
    weights = np.random.random(100)
    weights *= weights > 0.5
    counts = update_mask(weights, mask)
    assert counts == np.sum(mask)
    assert counts == np.sum(weights != 0)
    assert not np.any(mask[weights == 0])
    assert np.all(mask[weights != 0])


def test_update_mask_non_finite_values():
    mask = np.full(100, True)
    weights = np.random.random(100)
    weights[weights > 0.5] = np.nan
    counts = update_mask(weights, mask)
    assert counts == np.sum(mask)
    assert counts == np.sum(np.isfinite(weights))
    assert not np.any(mask[np.isnan(weights)])
    assert np.all(mask[np.isfinite(weights)])
