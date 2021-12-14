# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_polynomial import \
    ResamplePolynomial

import numpy as np
import pytest


@pytest.fixture
def data_2d():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.full(100, 1.0)
    error = np.full(100, 0.05)
    return coordinates, data, error


def test_adaptive_settings(data_2d):
    coordinates, data, error = data_2d

    r = ResamplePolynomial(coordinates, data)
    with pytest.raises(ValueError) as err:
        r.reduction_settings(adaptive_algorithm='shaped',
                             adaptive_threshold=1.0)
    assert "errors must be provided" in str(err.value).lower()

    r = ResamplePolynomial(coordinates, data, error=error)
    with pytest.raises(ValueError) as err:
        r.reduction_settings(adaptive_algorithm='shaped',
                             adaptive_threshold=1.0,
                             error_weighting=False)
    assert "error weighting must be enabled" in str(err.value).lower()

    r = ResamplePolynomial(coordinates, data, error=error, order=0)
    with pytest.raises(ValueError) as err:
        r.reduction_settings(adaptive_algorithm='shaped',
                             adaptive_threshold=1.0)
    assert "cannot be applied for polynomial fit of zero order" in str(
        err.value).lower()

    r = ResamplePolynomial(coordinates, data, error=error)
    with pytest.raises(ValueError) as err:
        r.reduction_settings(adaptive_algorithm='foo',
                             adaptive_threshold=1.0)
    assert "adaptive algorithm must be one of" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        r.reduction_settings(adaptive_algorithm='shaped',
                             adaptive_threshold=[1.0, 1.0, 1.0])
    assert "adaptive smoothing size does not match" in str(err.value).lower()

    # Testing valid adaptive smoothing
    r = ResamplePolynomial(coordinates, data, error=error, order=2)
    for algorithm in ['shaped', 'scaled']:
        settings = r.reduction_settings(adaptive_algorithm=algorithm,
                                        adaptive_threshold=1.0)
        assert np.allclose(settings['adaptive_threshold'], [1.0, 1.0])
        assert settings['adaptive_alpha'].shape == (0, 0, 0, 0)
        assert settings['alpha'].shape == (1,)
        assert settings['alpha'][0] == 0
        if algorithm == 'shaped':
            assert settings['shaped']
        else:
            assert not settings['shaped']

    # Testing no adaptive smoothing
    settings = r.reduction_settings(adaptive_algorithm=None,
                                    adaptive_threshold=1.0)
    assert not settings['shaped']
    assert settings['adaptive_alpha'].shape == (0, 0, 0, 0)
    assert settings['adaptive_threshold'] == 0

    settings = r.reduction_settings(adaptive_algorithm='shaped',
                                    adaptive_threshold=[0.0, 0.0])
    assert not settings['shaped']
    assert settings['adaptive_alpha'].shape == (0, 0, 0, 0)
    assert settings['adaptive_threshold'] == 0

    # Testing shaped is not enabled for less than 2 dimensions
    r = ResamplePolynomial(coordinates[0], data, error=error)
    settings = r.reduction_settings(adaptive_algorithm='shaped',
                                    adaptive_threshold=1.0)
    assert np.allclose(settings['adaptive_threshold'], [1.0])
    assert not settings['shaped']


def test_distance_weighting_settings(data_2d):

    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error, window=2.0)

    # Testing default smoothing in case adaptive is enabled without smoothing
    settings = r.reduction_settings(adaptive_algorithm='shaped',
                                    adaptive_threshold=1.0,
                                    smoothing=None,
                                    relative_smooth=True)
    assert settings['distance_weighting']
    assert settings['alpha'] == 1 / 3

    settings = r.reduction_settings(adaptive_algorithm='shaped',
                                    adaptive_threshold=1.0,
                                    smoothing=None,
                                    relative_smooth=False)

    assert settings['alpha'] == 1 / 3  # invariant for adaptive

    settings = r.reduction_settings(smoothing=None)
    assert not settings['distance_weighting']
    assert np.allclose(settings['alpha'], [0])

    # should be equal to 2 * smoothing^2 / window^2
    settings = r.reduction_settings(smoothing=0.25, relative_smooth=False)
    assert np.allclose(settings['alpha'], 2 * (0.25 ** 2) / 4)

    settings = r.reduction_settings(smoothing=0.25, relative_smooth=True)
    assert np.allclose(settings['alpha'], 0.25)

    settings = r.reduction_settings(smoothing=[0.25, 0.5],
                                    relative_smooth=False)
    assert np.allclose(settings['alpha'], [0.03125, 0.125])

    settings = r.reduction_settings(smoothing=[0.5, 0.5],
                                    relative_smooth=False)
    assert np.allclose(settings['alpha'], 0.125)
    # If equal across all dimensions, should use a faster single value.
    assert settings['alpha'].shape == (1,)

    with pytest.raises(ValueError) as err:
        r.reduction_settings(smoothing=[1, 2, 3])
    assert "smoothing size does not match" in str(err.value).lower()


def test_error_weighting_settings(data_2d):
    coordinates, data, error = data_2d

    r = ResamplePolynomial(coordinates, data, error=error)
    settings = r.reduction_settings()
    assert settings['error_weighting']

    settings = r.reduction_settings(error_weighting=False)
    assert not settings['error_weighting']

    r = ResamplePolynomial(coordinates, data)
    settings = r.reduction_settings()
    assert not settings['error_weighting']


def test_order_settings(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, order=2)

    settings = r.reduction_settings()
    assert np.allclose(settings['order'], 2)
    assert settings['order'].shape == (1,)
    assert not settings['order_varies']
    assert settings['order_symmetry']
    assert settings['order_minimum_points'] == 9

    r = ResamplePolynomial(coordinates, data, order=2, fix_order=False)
    settings = r.reduction_settings()
    assert settings['order_symmetry']
    assert settings['order_varies']

    r = ResamplePolynomial(coordinates, data, order=[2, 2], fix_order=False)
    settings = r.reduction_settings()
    assert not settings['order_symmetry']
    assert not settings['order_varies']
    assert settings['order_minimum_points'] == 9

    r = ResamplePolynomial(coordinates, data, order=[2, 3])
    settings = r.reduction_settings()
    assert np.allclose(settings['order'], [2, 3])
    assert settings['order_minimum_points'] == 12

    with pytest.raises(ValueError) as err:
        r.reduction_settings(order_algorithm='foo')
    assert 'unknown order algorithm' in str(err.value).lower()

    seen = []
    for algorithm in [None, 'bounded', 'extrapolate', 'counts']:
        settings = r.reduction_settings(order_algorithm=algorithm)
        assert settings['order_algorithm'] == str(algorithm).lower()
        assert settings['order_algorithm_idx'] not in seen
        seen.append(settings['order_algorithm_idx'])


def test_covar_and_mean_fit_settings(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error, order=2)
    settings = r.reduction_settings(is_covar=True)
    assert settings['is_covar']
    assert settings['mean_fit']

    settings = r.reduction_settings(is_covar=False)
    assert not settings['is_covar']
    assert not settings['mean_fit']

    r = ResamplePolynomial(coordinates, data, error=error, order=0)
    settings = r.reduction_settings(is_covar=True)
    assert settings['is_covar']
    assert settings['mean_fit']

    settings = r.reduction_settings(is_covar=False)
    assert not settings['is_covar']
    assert settings['mean_fit']

    r = ResamplePolynomial(coordinates, data, error=error, order=[0, 0])
    settings = r.reduction_settings()
    assert not settings['mean_fit']
