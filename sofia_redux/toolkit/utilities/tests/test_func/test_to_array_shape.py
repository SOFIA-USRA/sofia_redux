# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.func import to_array_shape


def test_errors():
    assert to_array_shape('a', (2, 2), dtype=float) is None
    assert to_array_shape(np.empty((2, 3)), (2, 4)) is None
    assert to_array_shape(np.empty((1, 1, 1, 1)), (2, 2)) is None


def test_standard():
    result = to_array_shape(1, (2, 2), dtype=np.float64)
    assert to_array_shape(None, (2, 2)) is None
    assert result.shape == (2, 2) and np.allclose(result, 1)
    assert result.dtype == 'float64'

    y, x = np.mgrid[:2, :3]
    z = x + y * 1.0
    result = to_array_shape(z, (2, 2, 3))
    assert result.shape == (2, 2, 3)
    assert np.allclose(result[0], z)
    assert np.allclose(result[1], z)


def test_scalar_shape():
    result = to_array_shape(1, 4, dtype=float)
    assert result.shape == (4,)
    assert np.allclose(result, 1.0)
    assert isinstance(result[0], float)
