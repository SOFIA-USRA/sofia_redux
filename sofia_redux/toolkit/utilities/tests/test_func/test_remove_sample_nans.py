# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.func import remove_sample_nans


def test_expected():
    samples = np.zeros((2, 10))
    samples[1, 1:3] = np.nan
    error = np.ones(10)
    error[8:] = 0
    expected_mask = [True, False, False, True, True,
                     True, True, True, False, False]

    mask = remove_sample_nans(samples, error, mask=True)
    assert np.allclose(mask, expected_mask)

    s, e = remove_sample_nans(samples, error)
    assert s.shape == (2, mask.sum())
    assert e.shape == (mask.sum(),)

    # test 1D
    samples = samples[1]
    mask = remove_sample_nans(samples, error, mask=True)
    assert np.allclose(mask, expected_mask)

    s, e = remove_sample_nans(samples, error)
    assert s.shape == (mask.sum(),)
    assert e.shape == (mask.sum(),)
