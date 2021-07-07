# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    evaluate_derivatives, polynomial_exponents, polynomial_terms,
    polynomial_derivative_map)

import numpy as np


def test_evaluate_derivatives():
    exponents = polynomial_exponents([1, 2])
    derivative_map = polynomial_derivative_map(exponents)

    c = np.random.random(exponents.size)  # coefficients
    points = np.random.random((2, 100))
    phi_points = polynomial_terms(points, exponents)

    derivatives = evaluate_derivatives(c, phi_points, derivative_map)

    # Equation is z = c_0 + c_1.x + c_2.y + c_3.x.y + c_4.y^2
    # dz/dy = c_2 + c_3.x + 2.c_4.y
    dy = c[2] + (c[3] * points[0]) + (2 * c[4] * points[1])

    # dz/dx = c_1 + c_3.y
    dx = c[1] + (c[3] * points[1])

    assert np.allclose(derivatives, [dx, dy])


def test_evaluate_derivatives_1d():
    exponents = polynomial_exponents(2)
    derivative_map = polynomial_derivative_map(exponents)
    c = np.random.random(exponents.size)  # coefficients
    points = np.random.random(100)[None]
    phi_point = polynomial_terms(points, exponents)

    # Equation is z = c_0 + c_1.x + c_2.x^2
    # dz/dx = c_1 + 2.c_2.x
    dx = c[1] + (2 * c[2] * points)
    derivative = evaluate_derivatives(c, phi_point, derivative_map)

    assert np.allclose(derivative, dx)
