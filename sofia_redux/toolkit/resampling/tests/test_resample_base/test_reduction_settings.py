# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.toolkit.resampling.resample_polynomial \
    import ResamplePolynomial


@pytest.fixture
def data_2d():
    coordinates = np.stack([x.ravel() for x in np.mgrid[:10, :10]])
    data = np.ones(coordinates.shape[1])
    error = np.ones(coordinates.shape[1])
    return coordinates, data, error


def test_edge_options(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error)

    with pytest.raises(ValueError) as err:
        r.reduction_settings(edge_threshold=[0.1, 0.2, 0.3])
    assert "edge threshold size does not match" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        r.reduction_settings(edge_threshold=0.5, edge_algorithm='foo')
    assert "unknown edge algorithm" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        r.reduction_settings(edge_threshold=[0.5, -0.5])
    assert "edge threshold must positive" in str(err.value).lower()

    for algorithm in ['ellipsoid', 'box', 'range']:
        with pytest.raises(ValueError) as err:
            r.reduction_settings(edge_threshold=1.5,
                                 edge_algorithm=algorithm)
        assert "edge threshold must be less than 1" in str(err.value).lower()

    with pytest.raises(ValueError) as err:
        r.reduction_settings(edge_threshold=np.inf,
                             edge_algorithm='distribution')
    assert "edge threshold must be less than inf" in str(err.value).lower()

    seen = []
    for algorithm in [None, 'distribution', 'ellipsoid', 'box', 'range']:
        settings = r.reduction_settings(edge_threshold=0.5,
                                        edge_algorithm=algorithm)
        assert settings['edge_algorithm'] == str(algorithm).lower()
        assert settings['edge_algorithm_idx'] not in seen
        seen.append(settings['edge_algorithm_idx'])
        assert np.allclose(settings['edge_threshold'], [0.5, 0.5])


def test_fit_threshold(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error, order=2)
    settings = r.reduction_settings()
    assert settings['fit_threshold'] == 0.0

    settings = r.reduction_settings(fit_threshold=3)
    assert settings['fit_threshold'] == 3.0


def test_parallel_settings(data_2d):
    coordinates, data, error = data_2d
    r = ResamplePolynomial(coordinates, data, error=error, order=2)
    settings = r.reduction_settings(use_processes=False, use_threading=False)
    assert settings['use_processes']
    assert not settings['use_threading']
    with pytest.raises(ValueError) as err:
        r.reduction_settings(use_processes=True, use_threading=True)
    assert "not both" in str(err.value)
    settings = r.reduction_settings(use_processes=True, use_threading=False)
    assert settings['use_processes']
    assert not settings['use_threading']
    settings = r.reduction_settings(use_processes=False, use_threading=True)
    assert not settings['use_processes']
    assert settings['use_threading']
