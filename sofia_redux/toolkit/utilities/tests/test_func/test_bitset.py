# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.utilities.func import bitset


def test_expected_results():
    arr = [1, 2, 3, 4, 5]
    assert (bitset(arr, 0) == np.array([1, 0, 1, 0, 1])).all()
    assert (bitset(arr, 1) == np.array([0, 1, 1, 0, 0])).all()
    assert (bitset(arr, [1, 2]) == np.array([0, 1, 1, 1, 1])).all()


def test_skip_checks():
    with pytest.raises(AttributeError):
        bitset([1, 2], 2, skip_checks=True)


def test_scalar():
    assert np.allclose(bitset(2, 1), 1)
