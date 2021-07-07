# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample import Resample

import numpy as np
import pytest


@pytest.fixture
def inputs():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    return coordinates, data


def test_features(inputs):
    coordinates, data = inputs
    r = Resample(coordinates, data)
    assert r.features == 2


def test_multiset(inputs):
    coordinates, data = inputs
    assert not Resample(coordinates, data).multi_set

    d2 = np.stack([data, data])
    assert Resample(coordinates, d2).multi_set


def test_n_sets(inputs):
    coordinates, data = inputs
    assert Resample(coordinates, data).n_sets == 1

    d2 = np.stack([data, data])
    assert Resample(coordinates, d2).n_sets == 2


def test_n_samples(inputs):
    coordinates, data = inputs
    assert Resample(coordinates, data).n_samples == 100


def test_window(inputs):
    coordinates, data = inputs
    w = Resample(coordinates, data, window=2).window
    assert w.shape == (2,)
    assert np.allclose(w, 2.0)


def test_order(inputs):
    coordinates, data = inputs
    o = Resample(coordinates, data, order=2).order
    assert o == 2
    o = Resample(coordinates, data, order=(1, 2)).order
    assert o.shape == (2,)
    assert np.allclose(o, [1, 2])


def test_fit_settings(inputs):
    coordinates, data = inputs
    r = Resample(coordinates, data)
    assert r.fit_settings is None

    # Fit settings are generated during __call__
    fit = r(5, 5)
    assert np.isclose(fit, 1)
    assert isinstance(r.fit_settings, dict)
