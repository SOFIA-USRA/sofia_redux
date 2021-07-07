# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample import Resample

import numpy as np


def test_scale_to_window():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:100, :100]])
    data = np.ones(coordinates.shape[1])
    r = Resample(coordinates, data, order=2)

    c_prime = r._scale_to_window(coordinates, radius=1)
    assert np.allclose(coordinates, c_prime)
    assert np.allclose(r._scale_offsets, [0, 0])
    assert np.allclose(r._radius, [1, 1])

    c_prime = r._scale_to_window(coordinates + 2, radius=[1, 1])
    assert np.allclose(coordinates, c_prime)
    assert np.allclose(r._scale_offsets, [2, 2])
    assert np.allclose(r._radius, [1, 1])

    c_prime = r._scale_to_window(coordinates, radius=2)
    assert np.allclose(coordinates, c_prime * 2)
    assert np.allclose(r._radius, [2, 2])

    c_prime = r._scale_to_window(coordinates, order=1)
    assert not np.allclose(coordinates, c_prime)
    assert np.allclose(c_prime * r._radius[:, None], coordinates)
