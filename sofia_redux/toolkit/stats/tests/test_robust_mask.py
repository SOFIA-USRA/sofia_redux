# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.stats.stats import robust_mask


def test_robust_mask():
    random = np.random.RandomState(41)
    # testing mask propagates
    data = random.rand(64, 64)
    test_mask = data < 0.9
    assert np.allclose(robust_mask(data, 3, mask=test_mask),
                       test_mask)

    # Testing zero threshold
    dnan = data.copy()
    dnan[~test_mask] = np.nan
    assert np.allclose(robust_mask(data, 0, mask=test_mask),
                       test_mask)

    # testing algorithm
    point_anomaly = data.copy()
    idx = np.nonzero(test_mask)
    point_anomaly[idx[0][0], idx[1][0]] = 1e3
    assert (robust_mask(point_anomaly, 3, mask=test_mask).sum()
            == (test_mask.sum() - 1))

    # test axis
    row_anomaly = data.copy()
    row_anomaly[32] += 1e3
    assert (robust_mask(row_anomaly, 3, axis=0).sum()
            < robust_mask(row_anomaly, 3, axis=1).sum())

    with pytest.raises(ValueError):
        robust_mask(data, 3, test_mask[0])
