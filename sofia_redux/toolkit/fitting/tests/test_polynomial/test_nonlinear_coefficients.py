# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial \
    import nonlinear_coefficients, linear_equation


@pytest.fixture
def data():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([7., 8])
    alpha, beta = linear_equation(a, b)
    return alpha, beta


def test_expected(data):
    result = nonlinear_coefficients(*data)
    assert np.allclose(result, [-1, 2])


def test_mask(data):
    result = nonlinear_coefficients(*data, mask=[False, False])
    assert np.isnan(result).all()
    result = nonlinear_coefficients(*data, mask=[True, True])
    assert np.allclose(result, [-1, 2])


def test_error(data):
    result = nonlinear_coefficients(*data, error=[1.0, 1.0])
    assert np.allclose(result, [-1, 2])
