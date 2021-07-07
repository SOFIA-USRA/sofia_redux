# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate import interp_1d_point


@pytest.fixture
def xyset():
    return np.arange(10, dtype=float), np.arange(10) * 2.0


@pytest.fixture
def reverse_xyset():
    return np.arange(10, dtype=float)[::-1], np.arange(10)[::-1] * 2.0


def test_out_of_range(xyset):
    assert np.isnan(interp_1d_point(*xyset, -1.0))
    assert np.isnan(interp_1d_point(*xyset, 10.5))


def test_out_of_range_reverse(reverse_xyset):
    assert np.isnan(interp_1d_point(*reverse_xyset, -1.0))
    assert np.isnan(interp_1d_point(*reverse_xyset, 10.5))


def test_on_point(xyset):
    assert interp_1d_point(*xyset, 0.0) == 0
    assert interp_1d_point(*xyset, 9.0) == 18


def test_on_point_reverse(reverse_xyset):
    assert interp_1d_point(*reverse_xyset, 0.0) == 0
    assert interp_1d_point(*reverse_xyset, 9.0) == 18


def test_min_point(xyset):
    assert interp_1d_point(*xyset, 3.5) == 7


def test_min_point_reverse(reverse_xyset):
    assert interp_1d_point(*reverse_xyset, 3.5) == 7
