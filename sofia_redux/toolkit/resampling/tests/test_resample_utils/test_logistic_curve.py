# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    logistic_curve, sigmoid)

import numpy as np


def test_logistic_curve():

    # Test the default is a plain sigmoid.
    x = np.linspace(-10, 10, 100)
    assert np.allclose(logistic_curve(x), sigmoid(x))

    # Test documentation agrees with result.
    rand = np.random.RandomState(42)
    x, x0, k, a, c, q, b, v = rand.random(8)
    y = logistic_curve(x, x0=x0, k=k, a=a, c=c, q=q, b=b, v=v)
    expected = a + ((k - a) / (c + (q * np.exp(-b * (x - x0)))) ** (1 / v))
    assert np.isclose(y, expected)
