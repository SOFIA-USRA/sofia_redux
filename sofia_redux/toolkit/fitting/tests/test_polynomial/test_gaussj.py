# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import gaussj


@pytest.fixture
def data():
    alpha = np.array([[1, 5, 2], [-1, -4, 1], [1, 3, 1]]).T.astype(float)
    beta = np.array([[-4, -12, 11], [15, 56, 13]]).T.astype(float)
    # solution = np.array([[3, 6, -1], [4, -3, 8]]).T
    return alpha, beta


def test_failure(data):
    alpha, beta = data
    with pytest.raises(ValueError):
        gaussj(alpha[0], beta)

    with pytest.raises(ValueError):
        gaussj(alpha, np.ones((2, 2)))

    with pytest.raises(ValueError):
        gaussj(alpha[0], np.arange(10))


def test_success(data):
    alpha, beta = data
    x = gaussj(alpha, beta)
    assert np.allclose(alpha @ x, beta)
    x, ai = gaussj(alpha, beta, invert=True)
    assert np.allclose(alpha @ x, beta)
    assert np.allclose(ai @ alpha, np.identity(alpha.shape[0]))


def test_singular(data):
    alpha, beta = data
    x = gaussj(alpha * 0, beta)
    assert x.shape == beta.shape and np.isnan(x).all()
    x, ai = gaussj(alpha * 0, beta, invert=True)
    assert x.shape == beta.shape and np.isnan(x).all()
    assert ai.shape == alpha.shape and np.isnan(ai).all()
