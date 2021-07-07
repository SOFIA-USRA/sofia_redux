# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.resampling.resample_utils import (
    solve_coefficients, solve_amat_beta, polynomial_exponents,
    polynomial_terms)


def test_solve_coefficients():

    data = (2.0 * np.arange(10)) + 1.0
    coordinates = np.arange(10, dtype=np.float64)[None]
    weights = np.ones(10)
    exponents = polynomial_exponents(1, ndim=1)
    x = polynomial_terms(coordinates, exponents)
    amat, beta = solve_amat_beta(x, data, weights)

    rank, coefficients = solve_coefficients(amat, beta)
    assert rank == 2
    assert np.allclose(coefficients, [1, 2])
