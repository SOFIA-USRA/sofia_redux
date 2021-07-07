# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.interpflagspec import interpflagspec
import pytest


@pytest.fixture
def data():
    x = np.arange(10).astype(float)
    y = np.arange(10) // 3
    return x, y


def test_invalid_input(data):
    x, y = data
    with pytest.raises(ValueError) as err:
        interpflagspec(x, y[:-1], 5)
    assert str(err.value) == 'x and y array shape mismatch'
    with pytest.raises(ValueError) as err:
        interpflagspec(x, np.full(x.shape, 'a'), 5)
    assert 'y must be convertable to' in str(err.value)
    assert np.allclose(interpflagspec(x * np.nan, y, [1, 2], cval=-1), -1)
    assert np.allclose(interpflagspec(x, y, [100, 101], cval=-1), -1)


def test_expected_output(data):
    x, y = data
    assert np.allclose(interpflagspec(x, y, x), y)
    assert np.allclose(interpflagspec(x, y, x + 0.5),
                       [0, 0, 1, 1, 1, 3, 2, 2, 3, 0])
    assert np.allclose(interpflagspec(x, y, x + 0.5, nbits=1),
                       [0, 0, 1, 1, 1, 1, 0, 0, 1, 0])

    assert np.allclose(interpflagspec(x, y, x[2] + 0.5), 1)
