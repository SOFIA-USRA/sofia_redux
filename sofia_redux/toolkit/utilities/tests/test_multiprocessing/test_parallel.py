# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing \
    import _parallel, in_windows_os

import numpy as np
import psutil
import pytest

try:
    from joblib import delayed, Parallel
    assert delayed
    assert Parallel
    have_joblib = True
except ImportError:  # pragma: no cover
    have_joblib = False


def adder(args, i):
    x, y = args
    return x[i] + y[i]


@pytest.fixture
def adder_data():
    n = 100
    xy = np.arange(n)
    xy = np.vstack((xy, xy + 100))
    expected = np.sum(xy, axis=0)
    iterable = list(range(n))
    return xy, iterable, expected


def test_errors(adder_data):
    args, iterable, expected = adder_data
    kwargs = None
    jobs = 2
    with pytest.raises(ValueError) as err:
        _parallel(jobs, adder, args, kwargs, iterable, force_processes=True,
                  force_threading=True)
    assert "Can either force threading or processes, not both." in str(
        err.value)

    with pytest.raises(NotImplementedError) as err:
        _parallel(jobs, adder, args, kwargs, iterable, package='foo')
    assert "foo package is not supported" in str(err.value)


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_default_settings(adder_data):
    args, iterable, expected = adder_data
    result = _parallel(2, adder, args, None, iterable)
    assert np.allclose(result, expected)


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_serial_processing(adder_data):
    args, iterable, expected = adder_data
    kwargs = None
    for jobs in [None, -1, 0, 1]:
        result = _parallel(jobs, adder, args, kwargs, iterable)
        assert np.allclose(result, expected)

    skip = np.full(len(iterable), True)
    skip[0] = False
    result = _parallel(2, adder, args, kwargs, iterable, skip=skip)
    assert len(result) == 1 and result[0] == expected[0]


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
@pytest.mark.skipif(not have_joblib, reason='Require joblib')
@pytest.mark.skipif(in_windows_os(), reason='Require joblib')
def test_joblib(adder_data):
    args, iterable, expected = adder_data
    result = _parallel(2, adder, args, None, iterable, package='joblib',
                       force_processes=True)
    assert np.allclose(result, expected)
    result = _parallel(2, adder, args, None, iterable, package='joblib',
                       force_threading=True)
    assert np.allclose(result, expected)
    result = _parallel(2, adder, args, None, iterable, package='joblib')
    assert np.allclose(result, expected)


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_multiprocessing(adder_data):
    args, iterable, expected = adder_data
    result = _parallel(2, adder, args, None, iterable,
                       package='multiprocessing', force_processes=True)
    assert np.allclose(result, expected)
    result = _parallel(2, adder, args, None, iterable,
                       package='multiprocessing', force_threading=True)
    assert np.allclose(result, expected)
    result = _parallel(2, adder, args, None, iterable,
                       package='multiprocessing')
    assert np.allclose(result, expected)
