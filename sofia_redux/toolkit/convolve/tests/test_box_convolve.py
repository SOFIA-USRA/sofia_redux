# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.convolve.kernel import BoxConvolve, convolve


def test_expected():
    image = np.zeros((100, 100))
    image[50, 50] = 1
    s = BoxConvolve(image, 3, normalize=False)
    mask = np.full(s.result.shape, False)
    mask[49:52, 49:52] = True
    assert np.allclose(s.result[mask], 1)
    assert np.allclose(s.result[~mask], 0)

    # also test convolve wrapper for this case
    wspec = convolve(image, 3, normalize=False)
    assert np.allclose(s.result, wspec)
