# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import polynomial_exponents

import numpy as np
import pytest


def test_polynomial_exponents():
    assert np.allclose(polynomial_exponents(2),
                       [[0], [1], [2]])

    assert np.allclose(polynomial_exponents(2, ndim=2),
                       [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [0, 2]])

    assert np.allclose(polynomial_exponents([1, 2]),
                       [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2]])

    assert np.allclose(polynomial_exponents([1, 2], use_max_order=True),
                       [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [0, 2]])


def test_error():
    with pytest.raises(ValueError) as err:
        polynomial_exponents([[1, 2]])

    assert "order must have 0 or 1 dimensions" \
        in str(err.value).lower().strip()
