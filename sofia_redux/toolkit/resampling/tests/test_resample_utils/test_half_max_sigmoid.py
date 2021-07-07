# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    half_max_sigmoid, sigmoid)

import numpy as np


def test_half_max_sigmoid():

    # Test default is sigmoid
    x = np.linspace(-10, 10, 100)
    assert np.allclose(sigmoid(x), half_max_sigmoid(x))

    x_half, c, q, b, v = np.random.random(5)

    result = half_max_sigmoid(
        x_half, x_half=x_half, a=0, k=1, c=c, q=q, b=b, v=v)
    assert np.isclose(result, 0.5)

    # Move midpoint and check
    result = half_max_sigmoid(
        x_half, x_half=x_half, a=2, k=4, c=c, q=q, b=b, v=v)
    assert np.isclose(result, 3)
