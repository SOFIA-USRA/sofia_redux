# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.stats.stats import medcomb


def test_invalid_input():
    with pytest.raises(ValueError):
        medcomb([1, 2, 3], mask=[1, 2])

    with pytest.raises(ValueError):
        medcomb([1, 2, 3], variance=[1, 2])


def test_expected_dimension_output():
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    r = medcomb(data)
    assert r[0] == 1
    assert np.isclose(r[1], 0.244, atol=1e-3)
    r = medcomb(data, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], 0.733, atol=1e-3)


def test_nan_single_output():
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    data[2] = np.nan
    r = medcomb(data)
    assert r[0] == 0.5
    assert np.isclose(r[1], 0.092, atol=1e-3)
    r = medcomb(data, ignorenans=False)
    assert np.isnan(r[0])
    assert r[1] == 0


def test_nan_axis_output():
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    data[1, 1] = np.nan
    r = medcomb(data, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], [0.733, 1.1, 0.733], atol=1e-3)
    r = medcomb(data, axis=0, ignorenans=False)
    assert np.allclose(np.isnan(r[0]), [False, True, False])
    assert r[0][0] == 1 and r[0][2] == 1
    assert np.allclose(r[1], [0.733, 0, 0.733], atol=1e-3)


def test_mask_output():
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    mask = np.full(data.shape, True)
    mask[2] = False
    r = medcomb(data, mask=mask)
    assert r[0] == 0.5
    assert np.isclose(r[1], 0.092, atol=1e-3)


def test_invalid_data():
    data = np.full((3, 3), np.nan)
    r = medcomb(data, axis=0)
    assert np.isnan(r[0]).all()
    assert np.allclose(r[1], 0)


def test_datavar():
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    datavar = np.full_like(data, 2)
    r = medcomb(data, variance=datavar)
    assert r[0] == 1 and np.isclose(r[1], 0.349, atol=1e-3)
    r = medcomb(data, variance=datavar, axis=0)
    assert np.allclose(r[0], 1) and np.allclose(r[1], 1.047, atol=1e-3)
    data[1, 1] = np.nan
    r = medcomb(data, variance=datavar, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], [1.047, 1.571, 1.047], atol=1e-3)
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    data[:, 2] = np.nan
    r = medcomb(data, variance=datavar, axis=0)
    assert np.allclose(np.isnan(r[0]), [False, False, True])
    assert r[1][2] == 0

    # check bad datavar values propagate to calculations
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    datavar[0, 1] = 0
    r = medcomb(data, variance=datavar, axis=0)
    assert np.allclose(r[0], [1, 1.5, 1])
    assert np.allclose(r[1], [1.047, 1.571, 1.047], atol=1e-3)

    r = medcomb(data, variance=2, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], 1.04719755)


def test_returned():
    data = np.zeros((3, 3)) + np.arange(3)[:, None]
    result = medcomb(data)
    assert len(result) == 2
    result = medcomb(data, returned=False)
    assert isinstance(result, float)

    mask = np.full(data.shape, False)
    result = medcomb(data, mask=mask)
    assert np.allclose(result, [np.nan, 0], equal_nan=True)
    result = medcomb(data, mask=mask, returned=False)
    assert np.isnan(result)
    result = medcomb(data, mask=mask, returned=True, axis=-1)
    assert len(result) == 2
    assert np.allclose(result[0], np.nan, equal_nan=True)
    assert np.allclose(result[1], 0)
    result = medcomb(data, mask=mask, returned=False, axis=-1)
    assert result.shape == (3,)
    assert np.allclose(result, np.nan, equal_nan=True)
