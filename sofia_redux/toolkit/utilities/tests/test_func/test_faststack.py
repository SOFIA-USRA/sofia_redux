# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.func import faststack


def test_standard():
    y, x = np.mgrid[:2, :3]
    result = faststack(x, y)
    assert result.shape == (2, y.size)
    assert np.issubdtype(np.int, result.dtype)
    assert np.allclose(result[0], np.arange(6) % 3)
    assert np.allclose(result[1], np.arange(6) // 3)
