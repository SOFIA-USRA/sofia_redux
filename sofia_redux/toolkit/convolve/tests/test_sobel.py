# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.convolve.filter import sobel


def test_expected():
    image = np.eye(4)
    result = sobel(image)
    assert np.allclose(result,
                       [[4, 4, 2, 0],
                        [4, 0, 4, 2],
                        [2, 4, 0, 4],
                        [0, 2, 4, 4]])


def test_axis():
    image = np.eye(4)
    result = sobel(image, axis=0)
    assert np.allclose(result,
                       [[-2, 1, 1, 0],
                        [-3, 0, 2, 1],
                        [-1, -2, 0, 3],
                        [0, -1, -1, 2]])
    result = sobel(image, axis=1)
    assert np.allclose(result,
                       [[-2, -3, -1, 0],
                        [1, 0, -2, -1],
                        [1, 2, 0, -1],
                        [0, 1, 3, 2]])


def test_pnorm():
    image = np.eye(4)

    with pytest.raises(ValueError) as err:
        sobel(image, pnorm=0)
    assert 'pnorm must not equal zero' in str(err)

    r2 = sobel(image, pnorm=2)
    assert np.allclose(r2, [[2.82842712, 3.16227766, 1.41421356, 0.],
                            [3.16227766, 0., 2.82842712, 1.41421356],
                            [1.41421356, 2.82842712, 0., 3.16227766],
                            [0., 1.41421356, 3.16227766, 2.82842712]])
