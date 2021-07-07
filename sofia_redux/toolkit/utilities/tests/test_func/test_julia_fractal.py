# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.func import julia_fractal


def test_standard():
    result = julia_fractal(100, 100, iterations=100, normalize=True)
    assert np.unique(result).size == 100
    assert result.max() == 1
    assert result.dtype == np.dtype('float64')
    result = julia_fractal(100, 100, iterations=100, normalize=False)
    assert result.max() == 99
