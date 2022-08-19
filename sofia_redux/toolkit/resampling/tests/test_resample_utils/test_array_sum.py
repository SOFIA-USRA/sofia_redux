# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import array_sum

import numpy as np


def test_array_sum():
    rand = np.random.RandomState(42)
    bool_array = rand.random(100) > 0.5
    assert array_sum(bool_array) == bool_array.sum()

    float_array = rand.random(100)
    assert np.isclose(array_sum(float_array), float_array.sum())

    int_array = (rand.random(100) * 100).astype(int)
    assert np.isclose(array_sum(int_array), int_array.sum())
