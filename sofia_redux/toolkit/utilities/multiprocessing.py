# Licensed under a 3-clause BSD style license - see LICENSE.rst

from functools import partial

from astropy import log
import numpy as np
import psutil
import threading


__all__ = ['get_core_number', 'multitask']


def get_core_number(cores=True):
    """
    Returns the maximum number of CPU cores available

    Parameters
    ----------
    cores : bool or int, optional
        If False, returns 1.  If True, returns the maximum number
        of cores available.  An integer specifies an upper limit
        on the maximum number of cores to return

    Returns
    -------
    cores : int
        The maximum number of cores to use for parallel processing
    """
    if not cores:
        return 1
    maxcores = psutil.cpu_count()
    if cores is True:
        return maxcores
    elif isinstance(cores, (int, float)):
        return int(np.clip(cores, 1, maxcores))
    else:
        return maxcores


def multitask(func, iterable, args, kwargs, jobs=None, skip=None):
    """
    Process a series of tasks in serial, or in parallel using joblib.

    `multitask` is used to run a function multiple times on a series of
    arguments.  Tasks may be run in series (default), or in parallel using
    multi-processing via the joblib package.

    If an error is encountered while attempting to process in parallel with
    joblib, an attempt will be made to process the tasks in series.

    The function to process multiple times (`func`) must take one of the
    following forms::

        1. result[i] = func(args, iterable[i])
        2. result[i] = func(args, kwargs, iterable[i])

    Here, the above "args" is the same as the `args` argument.  i.e., the
    full argument list.  Setting the argument `kwargs` to `None` implies
    that `func` takes form 1, while anything else sets multitask to assume
    the function is of form 2.

    Since this is a non-standard method of specifying a function, it is highly
    likely the user will have to define their own `func`.  For example, to
    use multitask to add ten to a series of numbers:

    >>> from sofia_redux.toolkit.utilities.multiprocessing import multitask
    >>> numbers = list(range(5))
    >>> numbers
    [0, 1, 2, 3, 4]
    >>> def add_ten(args, i):
    ...     return args[i] + 10
    >>> multitask(add_ten, range(len(numbers)), numbers, None)
    [10, 11, 12, 13, 14]

    In the above example, `iterable` has been set to range(len(numbers))
    indicating that multitask should supply `i` to `add_ten` multiple times
    (0, 1, 2, 3, 4).  Note that `kwargs` is explicitly set to None indicating
    that `add_ten` is of form 1.  While multitask may seem like overkill in
    this example, it can be highly adaptable for complex procedures.

    The `skip` parameter can be used to skip processing of certain tasks.
    For example:

    >>> skip = [False] * len(numbers)
    >>> skip[2] = True
    >>> multitask(add_ten, range(len(numbers)), numbers, None, skip=skip)
    [10, 11, 13, 14]

    By default, parallelization is managed with the loky backend.  If
    calling code is known to perform better with the threading backend,
    it should be called within the joblib parallel_backend context manager:

    >>> from joblib import parallel_backend
    >>> with parallel_backend('threading', n_jobs=2):
    ...    multitask(add_ten, range(len(numbers)), numbers, None)
    [10, 11, 12, 13, 14]


    Parameters
    ----------
    func : function
        The function to repeat multiple times on different sets of arguments.
        Must be of the form func(args, i) if `kwargs` is None, or
        func(args, kwargs, i) otherwise, where `i` is a member of `iterable`.
    iterable : iterable
        A Python object that can be iterated though such that each member
        can be passed to `func` as the final argument.
    args
        Anything that should be passed to `func` as the first argument.  The
        intended use is such that the output of `func` is
        result[i] = f(args[iterable[i]]).
    kwargs : None or anything
        If set to None, multitask assumes `func` is of the form func(args, i).
        Otherwise, multitask assumes `func` is of the form
        func(args, kwargs, i).
    jobs : int, optional
        If set to a positive integer, processes tasks in parallel using
        `jobs` threads.  If negative, sets the number of threads to the
        number of CPUs - jobs + 1.  i.e., if jobs = -1, multitask will use
        all available CPUs.
    skip : array_like of bool, optional
        Should be of len(`iterable`), where a value of True signifies that
        processing should be skipped for that task and omitted from the
        final result.

    Returns
    -------
    result : list
        The final output where result[i] = func(args, iterable[i]).  Will be
        of length len(`iterable`) if `skip` is None, otherwise
        len(iterable) - sum(skip).
    """

    if jobs in [None, 0, 1]:
        return _serial(func, args, kwargs, iterable, skip=skip)

    return _joblib_parallel(
        int(jobs), func, args, kwargs, iterable, skip=skip)


def _wrap_function(func, args, kwargs):
    """
    Helper function for multitask.

    Removes The requirement of supplying args and kwargs to func.
    """
    if kwargs is None:
        return partial(func, args)
    else:
        return partial(func, args, kwargs)


def _serial(func, args, kwargs, iterable, skip=None):
    """
    Helper function for multitask.

    Processes func in serial.
    """
    mfunc = _wrap_function(func, args, kwargs)
    result = []

    if skip is None:
        for thing in iterable:
            result.append(mfunc(thing))
    else:
        for (skipit, thing) in zip(skip, iterable):
            if skipit:
                continue
            else:
                result.append(mfunc(thing))

    return result


def _joblib_parallel(jobs, func, args, kwargs, iterable, skip=None):
    """
    Helper function for multitask.

    Processes func in parallel using joblib multi-processing.
    """
    try:
        from joblib import delayed, Parallel
    except ImportError:  # pragma: no cover
        log.warning("joblib is not installed: will process serially")
        return _serial(func, args, kwargs, iterable, skip=skip)

    if jobs in [0, 1]:  # pragma: no cover
        # this should be unreachable, since multitask directly
        # calls _serial for this case
        return _serial(func, args, kwargs, iterable, skip=skip)

    mfunc = _wrap_function(func, args, kwargs)
    result = []

    if skip is None:
        for thing in iterable:
            result.append(delayed(mfunc)(thing))
    else:
        for (skipit, thing) in zip(skip, iterable):
            if skipit:
                continue
            else:
                result.append(delayed(mfunc)(thing))

    # If called from a thread (such as the redux GUI), the loky/multiprocessing
    # backends will likely crash.  In this case, threading is the only workable
    # option.

    if threading.current_thread() == threading.main_thread():
        require = None
    else:  # pragma: no cover
        require = 'sharedmem'

    executor = Parallel(n_jobs=jobs, mmap_mode='r+', require=require)
    return executor(result)
