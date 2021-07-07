# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import sigmoid

import numpy as np


def test_sigmoid():
    x = np.linspace(-10, 10, 100)
    assert np.allclose(sigmoid(x), 1 / (1 + np.exp(-x)))


def test_offset():
    x = np.linspace(-10, 10, 100)
    assert np.allclose(sigmoid(x, offset=2), 1 / (1 + np.exp(-x + 2)))


def test_factor():
    x = np.linspace(-10, 10, 100)
    assert np.allclose(sigmoid(x, factor=2), 1 / (1 + np.exp(-2 * x)))
