# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import tabinv


def test_failure():
    x = [np.nan, np.nan, 1, 2, np.nan, 3, np.nan, np.nan]
    with pytest.raises(ValueError):
        tabinv(x, 1.5, fast=False)

    with pytest.raises(ValueError) as err:
        tabinv(np.zeros((3, 3)), 1.5)
    assert "array must have 1-dimension" in str(err.value).lower()


def test_fast():
    x = [np.nan, np.nan, 1, 2, np.nan, 3, np.nan, np.nan]
    assert tabinv(x, 1.5) == 2.5


def test_findidx():
    x = [np.nan, np.nan, 1, 2, 3, np.nan, np.nan]
    assert tabinv(x, 1.5, fast=False) == 2.5
