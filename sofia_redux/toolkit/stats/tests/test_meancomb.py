# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.stats.stats import meancomb


@pytest.fixture
def data():
    return np.zeros((3, 3)) + np.arange(3)[:, None]


def test_invalid_input():
    with pytest.raises(ValueError):
        meancomb([1, 2, 3], mask=[1, 2])

    with pytest.raises(ValueError):
        meancomb([1, 2, 3], variance=[1, 2])

    assert meancomb([1, 2, 3], mask=[False] * 3) == (np.nan, 0.0)
    assert meancomb(np.full((3, 3), np.nan)) == (np.nan, 0.0)
    r = meancomb(np.full((3, 3), np.nan), axis=0)
    assert np.isnan(r[0]).all()
    assert np.allclose(r[1], 0)

    r = meancomb(np.full((3, 3), np.nan), axis=0, returned=False)
    assert np.isnan(r).all()


def test_expected_output(data):
    d = data
    assert np.allclose(meancomb(d), [1, 0.0833], atol=1e-4)
    r = meancomb(d, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], 1 / 3)
    r = meancomb(d, axis=0, variance=d * 0 + 2)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], 2 / 3)


def test_nans(data):
    data[1, 1] = np.nan
    r = meancomb(data)
    assert r[0] == 1
    assert np.isclose(r[1], 0.107, atol=1e-3)
    r = meancomb(data, ignorenans=False)
    assert np.isnan(r[0])
    assert np.isnan(r[1])
    r = meancomb(data, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], [1 / 3, 1, 1 / 3])
    datavar = np.ones_like(data)
    r = meancomb(data, variance=datavar)
    assert r[0] == 1 and r[1] == 0.125
    r = meancomb(data, variance=datavar, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], [1 / 3, 0.5, 1 / 3])
    r = meancomb(data, variance=datavar, axis=0, ignorenans=False)
    assert np.isnan(r[0][1]) and r[0][0] == 1 and r[0][2] == 1
    assert np.allclose(r[1], [1 / 3, 0, 1 / 3])
    data[:, 1] = np.nan
    r = meancomb(data, variance=datavar, axis=0)
    assert np.allclose(r[0], [1, np.nan, 1], equal_nan=True)
    assert np.allclose(r[1], [1 / 3, 0, 1 / 3])


def test_mask(data):
    mask = np.full(data.shape, True)
    mask[1, 1] = False
    r = meancomb(data, mask=mask, axis=0)
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], [1 / 3, 1, 1 / 3])
    r = meancomb(data, mask=mask, variance=np.full_like(data, 1))
    assert np.isclose(r[0], 1)
    assert np.isclose(r[1], 0.125)


def test_robust(data):
    data[1, 1] = 1e6
    info = {}
    r = meancomb(data, robust=5, axis=0, info=info,
                 variance=np.full_like(data, 1))
    expected_mask = np.full((3, 3), True)
    expected_mask[1, 1] = False
    assert np.allclose(expected_mask, info['mask'])
    assert np.allclose(r[0], 1)
    assert np.allclose(r[1], [1 / 3, 0.5, 1 / 3])


def test_returned(data):
    r = meancomb(data, axis=0, returned=False)
    assert isinstance(r, np.ndarray)
    r = meancomb(data, returned=False)
    assert isinstance(r, float)
