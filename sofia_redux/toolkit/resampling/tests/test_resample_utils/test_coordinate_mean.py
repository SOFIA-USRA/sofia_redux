# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import coordinate_mean

import numpy as np


def test_coordinate_mean():
    coordinates = np.zeros((3, 10)) + np.arange(3)[:, None]
    means = coordinate_mean(coordinates)
    assert np.allclose(means, np.arange(3))

    mask = np.full(10, True)
    mask[5:] = False
    coordinates[:, ~mask] += 10
    means = coordinate_mean(coordinates)
    assert np.allclose(means, np.arange(3) + 5)

    means = coordinate_mean(coordinates, mask=mask)
    assert np.allclose(means, np.arange(3))
