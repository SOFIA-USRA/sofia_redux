# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial

import numpy as np
import itertools


def counts_in_window(coordinates, center, window):
    c = np.asarray(center)
    w = np.asarray(window)
    r = np.sqrt(np.sum(((coordinates - c[:, None]) / w[:, None]) ** 2, axis=0))
    return np.sum(r <= 1)


def test_estimate_1d_feature_windows():
    coordinates = np.arange(100, dtype=float)[None]
    r = ResamplePolynomial(coordinates, np.zeros(100), order=2)
    window = r.estimate_feature_windows(coordinates, oversample=1)
    center = np.asarray([50.0])
    # For 1-D data, 2nd order polynomial fit requires 3 samples
    assert counts_in_window(coordinates, center, window) == 3
    for dx in np.arange(0, 1.1, 0.1):
        assert counts_in_window(coordinates, center + dx, window) >= 3


def test_estimate_2d_feature_windows():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:100, :100]])
    r = ResamplePolynomial(coordinates, np.zeros(coordinates.shape[1]),
                           order=2)
    window = r.estimate_feature_windows(coordinates, oversample=1)
    # For 2-D data require at least 9 samples
    center = np.full(2, 50.0)
    offsets = np.arange(0, 1.1, 0.1)
    assert counts_in_window(coordinates, center, window) == 9
    for dx, dy in itertools.product(offsets, offsets):
        o = np.array([dx, dy])
        assert counts_in_window(coordinates, center + o, window) >= 9


def test_estimate_3d_feature_windows():
    # Starts getting a little funky for higher dimensions,
    # but should still work
    coordinates = np.stack([x.ravel() for x in np.mgrid[:25, :25, :25]])
    r = ResamplePolynomial(coordinates, np.zeros(coordinates.shape[1]),
                           order=2)
    window = r.estimate_feature_windows(coordinates, oversample=1)
    center = np.full(3, 12.0)
    offsets = np.arange(0, 1.1, 0.1)
    # Require at least 27 samples for 2nd order 3-D polynomial fit
    for dx, dy, dz in itertools.product(offsets, offsets, offsets):
        o = np.array([dx, dy, dz])
        assert counts_in_window(coordinates, center + o, window) >= 27


def test_oversampling():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:100, :100]]) / 5
    # 16 points required
    # Expect twice as many
    r = ResamplePolynomial(coordinates, np.zeros(coordinates.shape[1]),
                           order=3)
    window = r.estimate_feature_windows(coordinates, oversample=2)
    offsets = np.arange(0, 1.1, 0.1) / 5
    center = np.floor(np.median(coordinates, axis=1))
    for dx, dy in itertools.product(offsets, offsets):
        o = np.array([dx, dy])
        assert counts_in_window(coordinates, center + o, window) >= 32

    w1 = r.estimate_feature_windows(coordinates, oversample=1)
    c2 = counts_in_window(coordinates, center, window)
    c1 = counts_in_window(coordinates, center, w1)
    # high tolerance because of discrete sampling / low numbers
    assert np.isclose(c2 / c1, 2, atol=0.5)


def test_percentiles():
    rand = np.random.RandomState(0)
    coordinates = rand.normal(loc=0, scale=1, size=(2, 10000))
    r = ResamplePolynomial(coordinates, np.zeros(coordinates.shape[1]),
                           order=2)

    w100 = r.estimate_feature_windows(coordinates, percentile=100)
    w50 = r.estimate_feature_windows(coordinates, percentile=50)
    w25 = r.estimate_feature_windows(coordinates, percentile=25)
    assert np.all(w100 < w50)
    assert np.all(w50 < w25)


def test_feature_bins():
    rand = np.random.RandomState(0)
    coordinates = rand.normal(loc=0, scale=2, size=(2, 10000))
    r = ResamplePolynomial(coordinates, np.zeros(coordinates.shape[1]),
                           order=2)
    w10 = r.estimate_feature_windows(coordinates, feature_bins=10)
    w100 = r.estimate_feature_windows(coordinates, feature_bins=100)
    center = np.zeros(2)
    c10 = counts_in_window(coordinates, center, w10)
    c100 = counts_in_window(coordinates, center, w100)

    # This is near the center of the distribution (normal), so c10 should be
    # much less accurate than c100 here, but more accurate near the edges
    # 9 is the required number of samples
    assert c10 > 9
    assert c100 > 9
    assert c10 > c100

    # np.hypot(4,4) sigma out
    ce10 = counts_in_window(coordinates, center + 4, w10)
    ce100 = counts_in_window(coordinates, center + 4, w100)

    assert ce100 < 9
    assert ce10 > 9


def test_1_bin():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:100, :100]])
    r = ResamplePolynomial(coordinates, np.zeros(coordinates.shape[1]),
                           order=2)
    w = r.estimate_feature_windows(coordinates, feature_bins=1)
    center = np.full(2, 5.0)
    counts = counts_in_window(coordinates, center, w)
    assert counts >= 9
