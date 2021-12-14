# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import (
    _wrap_function, wrap_function, pickle_object)

from astropy import log
import numpy as np
import os
import pytest


def base_arg_func(args, i):
    x, y = args
    return x[i] + y[i]


def base_kwargs_func(args, kwargs, i):
    x, y = args
    multiply = kwargs.get('multiply')
    if not multiply:
        return x[i] + y[i]
    else:
        return x[i] * y[i]


def test_wrap_basic_function():
    x = np.arange(10)
    y = np.arange(10) + 10
    args = (x, y)
    kwargs = None
    func = _wrap_function(base_arg_func, args, kwargs)
    assert func(1) == 12
    assert func(2) == 14

    kwargs = {'multiply': True}
    func2 = _wrap_function(base_kwargs_func, args, kwargs)
    assert func2(1) == 11
    assert func2(2) == 24
    kwargs['multiply'] = False
    func3 = _wrap_function(base_kwargs_func, args, kwargs)
    assert func3(1) == 12
    assert func3(2) == 14

    # This should have impacted func2 as well since no copy is performed
    assert func2(2) == 14


def test_wrap_function(tmpdir):
    x = np.arange(10)
    y = np.arange(10) + 10
    args, kwargs = (x, y), None
    logging_directory = str(tmpdir.mkdir('test_wrap'))
    func, invalid_pickle_file = wrap_function(base_arg_func, args, kwargs)
    log_pickle_file = pickle_object(
        log, os.path.join(logging_directory, 'logger.p'))

    assert func(1) == 12
    assert invalid_pickle_file is None

    with pytest.raises(ValueError) as err:
        wrap_function(
            base_arg_func, args, kwargs=kwargs, logger=log,
            log_directory='_does_not_exist_')
    assert "valid log directory" in str(err.value)

    func, test_log = wrap_function(
        base_arg_func, args, kwargs=kwargs, logger=log,
        log_directory=logging_directory)
    assert func((1, 5)) == 12
    assert os.path.isfile(test_log)
    assert 'multitask_log_5.p' in os.listdir(logging_directory)

    func, test_log = wrap_function(
        base_arg_func, args, kwargs=kwargs, logger=log_pickle_file,
        log_directory=logging_directory)
    assert func((1, 6)) == 12
    assert os.path.isfile(test_log)
    assert 'multitask_log_6.p' in os.listdir(logging_directory)
