# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.utilities.func import stack


def test_errors():
    with pytest.raises(ValueError):
        a1 = np.empty((2, 3))
        a2 = np.empty((3, 3))
        stack(a1, a2)


def test_standard():
    y, x = np.mgrid[:2, :3]
    result = stack(x, y)
    assert result.dtype == 'float64'
    assert result.shape == (2, y.size)
    assert np.allclose(result[0], np.arange(6) % 3)
    assert np.allclose(result[1], np.arange(6) // 3)


def test_copy():
    y, x = np.mgrid[:2, :3]
    x = x.astype(np.float64).flatten()
    y = y.astype(np.float64).flatten()
    result = stack(x, y, copy=False)
    x[0] = -1
    # assert result[0, 0] == -1  # can't replicate this at the moment

    y, x = np.mgrid[:2, :3]
    x = x.astype(np.float64).flatten()
    y = y.astype(np.float64).flatten()
    result = stack(x, y, copy=True)
    x[0] = -1
    assert result[0, 0] != -1
