# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.utilities.multiprocessing \
    import get_core_number, multitask


def test_get_cores():
    maxcores = get_core_number()
    assert get_core_number(cores=False) == 1
    assert get_core_number(cores=1e6) == maxcores
    assert get_core_number(cores=-1) == 1
    assert get_core_number(cores='a') == maxcores


def test_multitask():

    def adder(xy, i):
        return xy[0, i] + xy[1, i]

    xy = np.arange(100)
    xy = np.vstack((xy, xy + 100))

    # test serial
    expected = np.sum(xy, axis=0)
    result = multitask(adder, range(xy.shape[1]), xy, None, jobs=None)
    assert np.allclose(result, expected)
    result = multitask(adder, range(xy.shape[1]), xy, None, jobs=1)
    assert np.allclose(result, expected)

    # test parallel
    result = multitask(adder, range(xy.shape[1]), xy, None, jobs=-1)
    assert np.allclose(result, expected)

    # Again with skip
    skip = np.full(xy.shape[1], False)
    skip[1] = True
    expected = np.delete(expected, 1)

    # test serial
    result = multitask(adder, range(xy.shape[1]), xy, None,
                       jobs=None, skip=skip)
    assert np.allclose(result, expected)
    result = multitask(adder, range(xy.shape[1]), xy, None,
                       jobs=1, skip=skip)
    assert np.allclose(result, expected)

    # test parallel
    result = multitask(adder, range(xy.shape[1]), xy, None,
                       jobs=-1, skip=skip)
    assert np.allclose(result, expected)

    # test kwargs
    def adder(xy, kwargs, i):
        offset = kwargs.get('offset', 0)
        return xy[0, i] + xy[1, i] + offset

    expected = np.sum(xy, axis=0) + 1
    kwargs = {'offset': 1}
    result = multitask(adder, range(xy.shape[1]), xy, kwargs)
    assert np.allclose(result, expected)


def test_multitask_backend(mocker):
    from joblib import parallel_backend
    from joblib.parallel import get_active_backend

    # patch the Parallel class with one that just reports its
    # current backend
    class ReportBackend(object):
        def __init__(self, *args, **kwargs):
            # test that prefer='threads' was not passed
            assert kwargs.get('prefer') != 'threads'

        def __call__(self, *args, **kwargs):
            test = str(type(get_active_backend()[0])).lower()
            return test, args, kwargs

    mocker.patch('joblib.Parallel', ReportBackend)

    # default backend is loky (multi-processing)
    def adder(xy, i):
        return xy[0, i] + xy[1, i]
    inp = np.arange(4)
    result = multitask(adder, inp, inp, None, jobs=2)
    assert 'loky' in result[0]

    # backend can be overridden with context manager
    with parallel_backend('threading'):
        result = multitask(adder, inp, inp, None, jobs=2)
    assert 'threading' in result[0]
