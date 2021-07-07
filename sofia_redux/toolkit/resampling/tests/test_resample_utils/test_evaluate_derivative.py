# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    evaluate_derivative, polynomial_exponents, polynomial_terms,
    polynomial_derivative_map)

import numpy as np


def test_evaluate_derivative():
    exponents = polynomial_exponents([1, 2])
    derivative_map = polynomial_derivative_map(exponents)

    c = np.random.random(exponents.size)  # coefficients
    point = np.random.random(2)
    phi_point = polynomial_terms(point, exponents)

    derivatives = evaluate_derivative(c, phi_point, derivative_map)

    # Equation is z = c_0 + c_1.x + c_2.y + c_3.x.y + c_4.y^2
    # dz/dx = c_1 + c_3.y
    dx = c[1] + (c[3] * point[1])
    # dz/dy = c_2 + c_3.x + 2.c_4.y
    dy = c[2] + (c[3] * point[0]) + (2 * c[4] * point[1])

    assert np.allclose(derivatives, [dx, dy])


def test_evaluate_derivative_1d():
    exponents = polynomial_exponents(2)
    derivative_map = polynomial_derivative_map(exponents)

    c = np.random.random(exponents.size)  # coefficients
    point = np.random.random(1)
    phi_point = polynomial_terms(point, exponents)

    # Equation is z = c_0 + c_1.x + c_2.x^2
    # dz/dx = c_1 + 2.c_2.x
    dx = c[1] + (2 * c[2] * point[0])
    derivative = evaluate_derivative(c, phi_point, derivative_map)

    assert np.allclose(derivative, dx)
