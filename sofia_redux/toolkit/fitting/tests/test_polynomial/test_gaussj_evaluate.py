# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import gaussj_evaluate


@pytest.fixture
def data():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([7., 8])
    out = np.array([3.5, 4.5])[None].T
    return a, b, out


def test_expected(data):
    result = gaussj_evaluate(*data)
    assert np.allclose(result, 5.5)

    # test datavec
    a, b, out = data
    result = gaussj_evaluate(a[None], b[None], out)
    assert np.allclose(result, 5.5)
    assert result.shape == (1, 1)
