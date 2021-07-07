# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    polynomial_exponents, polynomial_derivative_map)

import numpy as np


def test_polynomial_derivative_map():

    # z = c_0 + c_1.x + c_2.x^2 + c_3.y + c_4.x.y + c_5.y^2
    # phi = [0]    [1]     [2]       [3]      [4]       [5]
    exponents = polynomial_exponents(2, ndim=2)
    derivative_map = polynomial_derivative_map(exponents)

    # dz/dx = c_1 + 2.c_2.x + c_4.y
    assert np.allclose(derivative_map[0],
                       [[1, 2, 1], [1, 2, 4], [0, 1, 3]])

    # dz/dy = c_3 + c_4.x + 2.c_5.y
    assert np.allclose(derivative_map[1],
                       [[1, 1, 2], [3, 4, 5], [0, 1, 3]])

    # z = c_0 + c_1.x + c_2.y + c_3.x.y + c_4.y^2
    # phi = [0]    [1]     [2]      [3]       [4]
    exponents = polynomial_exponents([1, 2])
    derivative_map = polynomial_derivative_map(exponents)

    # dz/dx = c_1 + c_3.y
    assert np.allclose(derivative_map[0],
                       [[1, 1, 0], [1, 3, -1], [0, 2, -1]])

    # dz/dy = c_2 + c_3.x + 2.c_4.y
    assert np.allclose(derivative_map[1],
                       [[1, 1, 2], [2, 3, 4], [0, 1, 2]])
