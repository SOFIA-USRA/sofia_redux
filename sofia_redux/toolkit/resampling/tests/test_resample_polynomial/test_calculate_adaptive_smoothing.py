# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial

import numpy as np
import pytest


@pytest.fixture
def data_2d():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:11, :11]])
    data = np.ones(coordinates.shape[1])
    error = 0.05
    rand = np.random.RandomState(0)
    noise = rand.normal(loc=0, scale=error, size=coordinates.shape[1])
    data += noise
    return coordinates, data, error


def test_no_adaptive(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error)
    settings = r.reduction_settings(adaptive_threshold=0.0)
    r.calculate_adaptive_smoothing(settings)
    assert r.fit_settings['adaptive_threshold'] is None
    assert r.fit_settings['adaptive_alpha'].shape == (0, 0, 0, 0)


def test_scaled(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error)
    settings = r.reduction_settings(adaptive_threshold=1.0, smoothing=0.5,
                                    adaptive_algorithm='scaled')
    r.calculate_adaptive_smoothing(settings)
    a0 = r.fit_settings['adaptive_alpha'].copy()
    assert a0.shape == (121, 1, 1, 2)

    settings = r.reduction_settings(adaptive_threshold=1.0, smoothing=0.5,
                                    adaptive_algorithm='scaled',
                                    relative_smooth=True)
    r.calculate_adaptive_smoothing(settings)
    a1 = r.fit_settings['adaptive_alpha'].copy()
    assert a1.shape == (121, 1, 1, 2)
    assert not np.allclose(a0, a1, equal_nan=True)


def test_shaped(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error)
    settings = r.reduction_settings(adaptive_threshold=1.0, smoothing=0.5,
                                    adaptive_algorithm='shaped')
    r.calculate_adaptive_smoothing(settings)
    assert r.fit_settings['adaptive_alpha'].shape == (121, 1, 2, 2)
