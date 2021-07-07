# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample import Resample

import numpy as np
import pytest


@pytest.fixture
def data_2d():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:11, :11]])
    data = np.full(121, 1.0)
    rand = np.random.RandomState(0)
    noise = rand.normal(loc=0, scale=0.05, size=121)
    error = np.full(121, 0.05)
    data += noise
    mask = np.full(121, True)
    mask[:50] = False
    return coordinates, data, error, mask


def test_basic_init(data_2d):
    coordinates, data, error, mask = data_2d
    r = Resample(coordinates, data)
    assert r.error.size == 0
    assert r.mask.sum() == 121
    assert r.order == 1
    assert r.window.size == 2
    assert r.sample_tree.phi_terms.shape == (3, 121)


def test_error(data_2d):
    coordinates, data, error, mask = data_2d
    r = Resample(coordinates, data, error=error)
    assert r.error.size == 121 and np.allclose(r.error, 0.05)


def test_mask(data_2d):
    coordinates, data, error, mask = data_2d
    r = Resample(coordinates, data, mask=mask)
    assert r.mask.sum() == (121 - 50)


def test_order(data_2d):
    coordinates, data, error, mask = data_2d
    r = Resample(coordinates, data, order=2)
    assert r.order == 2
    assert not r.sample_tree.order_varies
    assert r.sample_tree.phi_terms.shape == (6, 121)

    r = Resample(coordinates, data, order=2, fix_order=False)
    assert r.order == 2
    assert r.sample_tree.order_varies
    assert r.sample_tree.phi_terms.shape == (10, 121)

    r = Resample(coordinates, data, order=[1, 2], fix_order=False)
    assert np.allclose(r.order, [1, 2])
    assert not r.sample_tree.order_varies
    assert r.sample_tree.phi_terms.shape == (5, 121)


def test_window(data_2d):
    coordinates, data, error, mask = data_2d
    r = Resample(coordinates, data, window=2)
    assert np.allclose(r.window, 2) and r.window.shape == (2,)
    assert np.allclose(r.sample_tree.coordinates, coordinates / 2)

    r = Resample(coordinates, data, window=[1, 2])
    assert np.allclose(r.sample_tree.coordinates[0], coordinates[0])
    assert np.allclose(r.sample_tree.coordinates[1], coordinates[1] / 2)


def test_robust(data_2d):
    coordinates, data, error, mask = data_2d
    bad_data = data.copy()
    bad_data[0] = 100
    r = Resample(coordinates, bad_data, robust=5)
    assert r.mask.sum() == 120
    assert not r.mask[0, 0]


def test_negthresh(data_2d):
    coordinates, data, error, mask = data_2d
    bad_data = data.copy()
    bad_data -= bad_data.mean()  # center around zero
    sigma = error[0]
    bad_data[0] = -4.5 * sigma
    r = Resample(coordinates, bad_data, negthresh=5)
    assert r.mask.sum() == 121
    r = Resample(coordinates, bad_data, negthresh=4)
    assert r.mask.sum() == 120 and not r.mask[0, 0]


def test_window_estimate(data_2d):
    coordinates, data, error, mask = data_2d

    # Test order is getting through to estimate
    w1_estimate = Resample(coordinates, data, order=1).window
    w2_estimate = Resample(coordinates, data, order=2).window
    assert np.all(w2_estimate > w1_estimate)

    # Test supplied window is valid
    assert np.allclose(Resample(coordinates, data, order=2,
                                window=1).window, 1)

    # Test bins makes it through
    w_1bin = Resample(coordinates, data, order=1,
                      window_estimate_bins=1).window
    assert not np.allclose(w_1bin, w1_estimate)

    # Test percentile makes it though
    w_max = Resample(coordinates, data, order=1,
                     window_estimate_percentile=100).window
    assert np.all(w_max < w1_estimate)

    # Test oversample makes it though
    w_oversampled = Resample(coordinates, data, order=1,
                             window_estimate_oversample=4).window
    assert np.all(w_oversampled > w1_estimate)


def test_distance_metrics():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    r = Resample(coordinates, data, window=5)
    center = np.full((2, 1), 1.0)
    i1, d1 = r.sample_tree.query_radius(center, return_distance=True)

    r_cheb = Resample(coordinates, data, window=2,
                      metric='chebyshev')
    i2, d2 = r_cheb.sample_tree.query_radius(center, return_distance=True)
    assert not np.allclose(i2[0].shape, i1[0].shape)
    assert np.allclose(np.unique(d2[0]), [0, 0.5, 1])
