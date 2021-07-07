# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.utilities.func import nantrim


@pytest.fixture
def data():
    test = np.zeros((11, 11))
    test[:2] = np.nan
    test[-3:] = np.nan
    test[:, :1] = np.nan
    test[:, -4:] = np.nan
    test[5, 5] = np.nan
    return test


def test_invalid_input():
    with pytest.raises(ValueError):
        nantrim([1, 2, 3], -1)

    with pytest.raises(ValueError):
        nantrim([1, 2, 3], 4)


def test_expected_output():
    n = np.nan
    mix = np.array([n, n, n, 1, 1, 1, n, 1, n, 1, n, n, n])
    fmix = np.isfinite(mix)
    none = np.full(5, np.nan)
    real = np.full(5, 1)
    fnone = np.isfinite(none)
    freal = np.isfinite(real)

    for i in range(4):
        assert np.allclose(nantrim(none, i), fnone)
        assert np.allclose(nantrim(real, i), freal)

    idx = nantrim(mix, 0)
    assert idx[:10].all()
    assert not idx[10:].any()

    idx = nantrim(mix, 1)
    assert not idx[:3].any()
    assert idx[3:].all()

    idx = nantrim(mix, 2)
    assert idx[3:10].all()
    assert not idx[:3].any()
    assert not idx[10:].any()

    idx = nantrim(mix, 3)
    assert np.allclose(idx, fmix)


def test_trim(data):
    test = data
    assert np.allclose(nantrim(test, 0, trim=True).shape, (8, 7))
    assert np.allclose(nantrim(test, 1, trim=True).shape, (9, 10))
    assert np.allclose(nantrim(test, 2, trim=True).shape, (6, 6))
    assert nantrim(test, 3, trim=True).shape == (35,)


def test_bounds(data):
    test = data
    assert np.allclose(nantrim(test, 0, bounds=True), [[0, 0], [8, 7]])
    assert np.allclose(nantrim(test, 1, bounds=True), [[2, 1], [11, 11]])
    assert np.allclose(nantrim(test, 2, bounds=True), [[2, 1], [8, 7]])
    assert np.allclose(nantrim(test * np.nan, 2, bounds=True),
                       [[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        nantrim(test, 3, bounds=True)
