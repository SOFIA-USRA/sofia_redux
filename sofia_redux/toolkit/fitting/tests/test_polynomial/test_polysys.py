# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.fitting.polynomial import polysys


@pytest.fixture
def data1d():
    x = np.arange(10, dtype=float)
    y = 1 + x * 2
    return np.array([x, y]).astype(float)


@pytest.fixture
def data2d():
    y, x = np.mgrid[:5, :5]
    z = x * y
    return np.array([x.ravel(), y.ravel(), z.ravel()]).astype(float)


@pytest.fixture
def data3d():
    # oooooOOOO fancy!
    z, y, x = np.mgrid[:4, :4, :4]
    data = x + y + z
    return np.array([a.ravel().astype(float) + i
                     for (i, a) in enumerate([x, y, z, data])])


def test_failure(data1d):
    samples = data1d
    with pytest.raises(ValueError):
        polysys(np.zeros((2, 2, 2)), 3)

    with pytest.raises(ValueError) as err:
        polysys(samples, 1, exponents=[[0, 1]])
    assert "exponents and samples features mismatch" in str(err.value).lower()


def test_1d(data1d):
    samples = data1d
    alpha, beta = polysys(samples, 0)
    assert np.isclose(alpha, 10) and np.isclose(beta, 100)
    alpha, beta = polysys(samples, 1)
    assert np.allclose(alpha, [[10, 45], [45, 285]])
    assert np.allclose(beta, [100, 615])


def test_1d_error(data1d):
    samples = data1d
    alpha, beta = polysys(samples, 1, error=2)
    assert np.allclose(alpha, [[2.5, 11.25], [11.25, 71.25]])
    assert np.allclose(beta, [25, 153.75])
    alpha, beta = polysys(samples, 1, error=[2] * samples.shape[1])
    assert np.allclose(alpha, [[2.5, 11.25], [11.25, 71.25]])
    assert np.allclose(beta, [25, 153.75])
    alpha, beta = polysys(samples, 1, error=np.full_like(samples[0], 2))
    assert np.allclose(alpha, [[2.5, 11.25], [11.25, 71.25]])
    assert np.allclose(beta, [25, 153.75])


def test_04_2d(data2d):
    samples = data2d
    alpha, beta = polysys(samples, [1, 1])
    assert np.allclose(alpha, [[25., 50., 50., 100.],
                               [50., 150., 100., 300.],
                               [50., 100., 150., 300.],
                               [100., 300., 300., 900.]])
    assert np.allclose(beta, [100, 300, 300, 900])


def test_05_2d_error(data2d):
    samples = data2d
    alpha, beta = polysys(samples, 1, error=2)
    assert np.allclose(alpha, [[6.25, 12.5, 12.5],
                               [12.5, 37.5, 25.],
                               [12.5, 25., 37.5]])
    assert np.allclose(beta, [25., 75., 75.])


def test_06_nd(data3d):
    # Now I'm just showing off
    samples = data3d
    alpha, beta = polysys(samples, 1)
    assert np.allclose(alpha, [[64., 96., 160., 224.],
                               [96., 224., 240., 336.],
                               [160., 240., 480., 560.],
                               [224., 336., 560., 864.]])
    assert np.allclose(beta, [480., 800., 1280., 1760.])


def test_07_nans(data2d):
    samples = data2d
    samples[0, 0] = np.nan
    alpha, beta = polysys(samples, [2, 2], ignorenans=True)
    assert not np.isnan(alpha).any() and not np.isnan(beta).any()
    alpha, beta = polysys(samples, [2, 2], ignorenans=False)
    assert np.isnan(alpha).any() and not np.isnan(beta).any()
    alpha, beta = polysys(samples * np.nan, [2, 2], ignorenans=False)
    # assert np.isnan(alpha).any() and np.isnan(beta).any()  # for np.nansum
    assert np.isnan(alpha).any() and np.allclose(beta, 0)  # for bn.nansum
    alpha, beta = polysys(samples * np.nan, [2, 2], ignorenans=True)
    assert np.isnan(alpha).all() and np.isnan(beta).all()
