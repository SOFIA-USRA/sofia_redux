# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import line_shift


@pytest.fixture
def line():
    return np.arange(100, dtype=float)


def test_invalid(line):
    result = line_shift(line, 200, order=1)
    assert np.isnan(result).all()
    result = line_shift(line, 1, order=-1)
    assert np.isnan(result).all()


def test_order0(line):
    result = line_shift(line.astype(int), 3, order=0)
    assert np.isnan(result[:3]).all()
    assert np.allclose(result[3:], np.arange(97))

    result = line_shift(line.astype(int), 3, order=0, missing=-1)
    assert np.issubdtype(result.dtype, np.integer)
    assert np.allclose(result[:3], -1)
    assert np.allclose(result[3:], np.arange(97))


def test_expected(line):
    result = line_shift(line.astype(int), 3, order=1)
    assert result.dtype == np.float32
    assert np.isnan(result[:3]).all()
    assert np.allclose(result[3:], np.arange(97))

    result = line_shift(line, 3, order=1)
    assert result.dtype == np.float64
    assert np.isnan(result[:3]).all()
    assert np.allclose(result[3:], np.arange(97))

    # test fractional shifts
    result = line_shift(line, 0.5, order=1)
    assert np.isnan(result[0])
    assert np.allclose(result[1:], np.arange(99) + 0.5)
