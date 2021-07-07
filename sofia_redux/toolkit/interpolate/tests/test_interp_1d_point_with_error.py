# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.interpolate.interpolate \
    import interp_1d_point_with_error as interp


@pytest.fixture
def xyeset():
    return (np.arange(10, dtype=float),
            np.arange(10) * 2.0,
            np.arange(10, dtype=float))


@pytest.fixture
def reverse_xyeset():
    return (np.arange(10, dtype=float)[::-1],
            np.arange(10)[::-1] * 2.0,
            np.arange(10, dtype=float)[::-1])


def test_out_of_range(xyeset):
    assert np.isnan(interp(*xyeset, -1.0)).all()
    assert np.isnan(interp(*xyeset, 10.5)).all()


def test_out_of_range_reverse(reverse_xyeset):
    assert np.isnan(interp(*reverse_xyeset, -1.0)).all()
    assert np.isnan(interp(*reverse_xyeset, 10.5)).all()


def test_on_point(xyeset):
    assert np.allclose(interp(*xyeset, 0.0), [0, 0])
    assert np.allclose(interp(*xyeset, 9.0), [18, 9])


def test_on_point_reverse(reverse_xyeset):
    assert np.allclose(interp(*reverse_xyeset, 0.0), [0, 0])
    assert np.allclose(interp(*reverse_xyeset, 9.0), [18, 9])


def test_min_point(xyeset):
    # Check both sides
    assert np.allclose(interp(*xyeset, 3.25), [6.5, 3.25])
    assert np.allclose(interp(*xyeset, 3.75),
                       [7.5, 4.190763653560053])


def test_min_point_reverse(reverse_xyeset):
    # Check both sides
    assert np.allclose(interp(*reverse_xyeset, 3.25), [6.5, 3.25])
    assert np.allclose(interp(*reverse_xyeset, 3.75),
                       [7.5, 4.190763653560053])
