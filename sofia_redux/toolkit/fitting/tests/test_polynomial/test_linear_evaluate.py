# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import linear_evaluate


@pytest.fixture
def data():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([7., 8])
    out = np.array([3.5, 4.5])[None].T
    return a, b, out


def test_disallow_errors(data):
    result = linear_evaluate(*data, allow_errors=False)
    assert np.allclose(result, 5.5)
    a, b, out = data

    # test datavec
    result = linear_evaluate(a[None], b[None], out, allow_errors=False)
    assert np.allclose(result, 5.5)
    assert result.shape == (1, 1)

    with pytest.raises(np.linalg.LinAlgError):
        linear_evaluate(a[None] * 0, b[None], out, allow_errors=False)


def test_allow_errors(data):
    result = linear_evaluate(*data, allow_errors=True)
    assert np.allclose(result, 5.5)
    a, b, out = data

    # test datavec
    result = linear_evaluate(a[None], b[None], out, allow_errors=True)
    assert np.allclose(result, 5.5)
    assert result.shape == (1, 1)

    result = linear_evaluate(a[None] * 0, b[None], out, allow_errors=True)
    assert np.isnan(result).all()

    a2 = np.stack((a, a))
    b2 = np.stack((b, b))
    a2[0] *= 0
    result = linear_evaluate(a2, b2, out, allow_errors=True)
    assert np.allclose(result, [[np.nan], [5.5]], equal_nan=True)

    # test no datavec with error
    result = linear_evaluate(a * 0, b, out, allow_errors=True)
    assert np.isnan(result).all()
