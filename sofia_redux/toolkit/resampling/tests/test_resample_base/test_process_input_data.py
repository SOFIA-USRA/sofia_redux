# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_base import ResampleBase

import numpy as np
import pytest


def test_data_only():
    data = np.ones(100)
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    r = ResampleBase(coordinates, data)
    r._process_input_data(data, coordinates)
    assert not r.multi_set
    assert r.n_sets == 1
    assert r.data.shape == (1, 100)
    assert r._valid_set.shape == (1,)
    assert np.all(r._valid_set)

    assert r.error.shape == (1, 0)
    assert not r._error_valid
    assert r.mask.shape == (1, 100)
    assert np.all(r.mask)

    with pytest.raises(ValueError) as err:
        r._process_input_data(data * np.nan, coordinates)
    assert "all data has been flagged" in str(err.value).lower()

    r._process_input_data([data, data], coordinates)
    assert r.multi_set
    assert r.data.shape == (2, 100)
    assert r.n_sets == 2


def test_robust():
    data = np.ones(100)
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    r = ResampleBase(coordinates, data)
    rand = np.random.RandomState(0)
    data += rand.normal(loc=0, scale=0.05, size=data.shape)
    data[0] += 10
    r._process_input_data(data, coordinates, robust=3)
    assert r.mask.sum() == 99
    assert not r.mask[0, 0]


def test_negthresh():
    data = np.ones(100)
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    rand = np.random.RandomState(0)
    data += rand.normal(loc=0, scale=0.05, size=data.shape)
    r = ResampleBase(coordinates, data)
    data[0] -= 10
    r._process_input_data(data, coordinates, negthresh=3)
    assert r.mask.sum() == 99
    assert not r.mask[0, 0]


def test_error_array():
    data = np.ones(100)
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    r = ResampleBase(coordinates, data)
    error = data / 2
    r._process_input_data(data, coordinates, error=error)
    assert r.error.shape == (1, 100)
    assert np.allclose(r.error, 0.5)

    data[0] = np.nan
    r._process_input_data(data, coordinates, error=error)
    assert np.sum(np.isfinite(r.error)) == 99
    assert np.isnan(r.error[0, 0])

    r._process_input_data(data, coordinates, error=0.5)
    assert r.error.shape == (1, 1)
    assert np.allclose(r.error, 0.5)

    with pytest.raises(ValueError) as err:
        r._process_input_data(data, coordinates, error=[0.5, 0.5])

    assert "Error must be a single value, an array" in str(err.value)

    # test multiple data sets
    data = np.stack([np.ones(100), np.ones(100)])
    r._process_input_data(data, coordinates, error=0.1)
    assert np.allclose(r.error, [[0.1], [0.1]])
    r._process_input_data(data, coordinates, error=[0.1, 0.2])
    assert np.allclose(r.error, [[0.1], [0.2]])


def test_mask():
    data = np.ones(100)
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    r = ResampleBase(coordinates, data)
    mask = np.full(100, True)
    mask[0] = False
    r._process_input_data(data, coordinates, mask=mask)
    assert np.sum(np.isfinite(r.data)) == 99
    assert np.isnan(r.data[0, 0])
