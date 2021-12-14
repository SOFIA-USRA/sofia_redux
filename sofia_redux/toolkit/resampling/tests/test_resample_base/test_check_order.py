# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.resample_polynomial \
    import ResamplePolynomial


def test_check_errors():
    with pytest.raises(ValueError) as err:
        ResamplePolynomial.check_order(np.empty((3, 3)), 2, 1000)
    assert "should be a scalar or 1-d array" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        ResamplePolynomial.check_order(np.empty(3), 2, 1000)
    assert "does not match number of features" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        ResamplePolynomial.check_order(3, 2, 0)
    assert "too few data samples" in str(err.value).lower()


def test_check_order():
    o = ResamplePolynomial.check_order(2, 2, 1000)
    assert o == 2

    o = ResamplePolynomial.check_order([2, 2], 2, 1000)
    assert o.shape == (2,)
    assert np.allclose(o, 2)
