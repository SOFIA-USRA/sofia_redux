# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import interp_error_1d


@pytest.fixture
def data():
    x = np.arange(10)
    error = x * 0 + 1.0
    return x, error


def test_invalid_input(data):
    x, error = data
    with pytest.raises(ValueError):
        interp_error_1d(x, error[:-1], 0.1)
    assert np.isnan(interp_error_1d(x, error, [-1, -2], cval=np.nan)).all()


def test_expected_output(data):
    x, error = data
    xout = np.linspace(3, 4, 11)
    assert np.allclose(
        interp_error_1d(x, error, xout),
        [1, 1.01, 1.04, 1.09, 1.15, 1.22, 1.15, 1.09, 1.04, 1.01, 1],
        atol=0.01)
    result = interp_error_1d(x, error, 5.5)
    assert isinstance(result, float)
    assert np.isclose(result, 1.22, atol=0.01)
