# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    convert_to_numba_list)

from numba.typed import List as numba_list
from numba import TypingError
import numpy as np

import pytest


def test_lists():
    x = [1, 2, 3]
    result = convert_to_numba_list(x)
    assert isinstance(result, numba_list)


def test_invalid_elements():
    x = [1, 2, 3, 4.0]  # cannot mix types inside list
    with pytest.raises(TypingError):
        convert_to_numba_list(x)


def test_numpy_arrays():
    x = (np.arange(3), np.arange(4, 10), np.empty(0, dtype=int))
    result = convert_to_numba_list(x)
    assert np.allclose(result[0], x[0])
    assert np.allclose(result[1], x[1])
    assert result[2].size == 0
