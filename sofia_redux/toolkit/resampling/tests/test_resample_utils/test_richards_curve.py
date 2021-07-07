# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    richards_curve, sigmoid)

import numpy as np


def test_richards_curve():
    x = np.linspace(-10, 10, 100)

    # Test the default is a plain sigmoid.
    assert np.allclose(richards_curve(x), sigmoid(x))

    # Test documentation agrees with result.
    x, q, a, k, b, x0 = np.random.random(6)

    y = richards_curve(x, q=q, a=a, k=k, b=b, x0=x0)
    expected = a + ((k - a) / (1 + (q * np.exp(-b * (x - x0)))) ** (1 / q))
    assert np.isclose(y, expected)
