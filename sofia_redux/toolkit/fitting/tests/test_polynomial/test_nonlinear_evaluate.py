# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.fitting.polynomial \
    import nonlinear_evaluate, linear_equation


def test_expected():
    a = np.array([[3, 4], [5, 6.]])
    b = np.array([7., 8])
    alpha, beta = linear_equation(a, b)
    out = np.array([3.5, 4.5])
    result = nonlinear_evaluate(alpha, beta, out)
    assert np.allclose(result, 5.5)
