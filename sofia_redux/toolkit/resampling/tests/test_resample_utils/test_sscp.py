# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import sscp
import numpy as np


def test_sscp():
    a = (np.arange(12, dtype=np.float64) + 1).reshape((3, 4))
    ata = sscp(a)

    expected = np.array([[30, 70, 110],
                         [70, 174, 278],
                         [110, 278, 446]])

    assert np.allclose(ata, expected)

    ata_norm = sscp(a, normalize=True)
    assert np.allclose(ata_norm, expected / a.shape[1])


def test_weighted_sscp():
    a = (np.arange(12, dtype=np.float64) + 1).reshape((3, 4))

    uniform_weights = np.full(a.shape[1], 1.0)

    expected_uniform = np.array([[30, 70, 110],
                                 [70, 174, 278],
                                 [110, 278, 446]])

    # Check returns the same for uniform weights
    result = sscp(a, weight=uniform_weights)
    assert np.allclose(result, expected_uniform)

    # Check result scales as weights^2
    result = sscp(a, weight=uniform_weights * 2)
    assert np.allclose(result, expected_uniform * (2 ** 2))

    # Check non uniform weighting results
    non_uniform_weights = np.array([0.5, 1.5, 1.5, 0.5])
    result = sscp(a, weight=non_uniform_weights)
    assert np.allclose(result,
                       [[33.5, 83.5, 133.5],
                        [83.5, 213.5, 343.5],
                        [133.5, 343.5, 553.5]])

    # Now do same with normalization
    result = sscp(a, weight=uniform_weights, normalize=True)
    assert np.allclose(result, expected_uniform / a.shape[1])

    # Scaling the weights should have no effect
    result = sscp(a, weight=uniform_weights * 2, normalize=True)
    assert np.allclose(result, expected_uniform / a.shape[1])

    result = sscp(a, weight=non_uniform_weights, normalize=True)
    assert np.allclose(result,
                       [[6.7, 16.7, 26.7],
                        [16.7, 42.7, 68.7],
                        [26.7, 68.7, 110.7]])
    # NOTE: This is the same result as above / 5 since sum(w^2) = 5
