# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial \
    import linear_equation, linear_vector_lstsq


@pytest.fixture
def data():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([[7., 8]])
    alpha, beta = linear_equation(a, b)
    return alpha, beta


def test_expected(data):
    alpha, beta = data
    result = linear_vector_lstsq(alpha, beta, np.array([[3.5, 4.5]]).T)
    assert np.allclose(result, 5.5)
    assert result.shape == (1, 1)
