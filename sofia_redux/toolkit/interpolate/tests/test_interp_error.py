# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import interp_error


def test_1d():
    x = np.arange(10)
    error = np.arange(10) * 1.0
    out = [2, 2.5]
    result = interp_error(x, error, out)
    assert np.allclose(result, [2, 2.6925824])
    result = interp_error(x[None].T, error, out)
    assert np.allclose(result, [2, 2.6925824])


def test_2d():
    y, x = np.mgrid[:10, :10]
    xy = np.stack([x.ravel() * 1.0, y.ravel()]).T
    error = np.ones(x.size)
    out = np.asarray([[1.0, 2.0], [1.5, 2.5]])
    result = interp_error(xy, error, out)
    assert np.allclose(result, [1, np.sqrt(3.5)])


def test_failure():
    with pytest.raises(ValueError) as err:
        interp_error(np.zeros((3, 3, 3)), np.arange(3), np.arange(3))
    assert "points must be a" in str(err.value).lower()
