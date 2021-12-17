# Licensed under a 3-clause BSD style license - see LICENSE.rst

from contextlib import contextmanager
from functools import partial
import logging
import multiprocessing as mp
import os
import regex
import shutil
import signal
import sys
import tempfile
import time
import threading

from astropy import log
import cloudpickle
import numpy as np
import psutil


__all__ = ['get_core_number', 'relative_cores', 'valid_relative_jobs',
           'multitask', 'pickle_object', 'unpickle_file', 'pickle_list',
           'unpickle_list', 'in_main_thread', 'log_with_multi_handler',
           'log_for_multitask', 'purge_multitask_logs', 'wrapped_with_logger',
           'log_records_to_pickle_file', 'MultitaskHandler', 'wrap_function']


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
    max_cores = psutil.cpu_count()
    if cores is True:
        return max_cores
    elif isinstance(cores, (int, float)):
        return int(np.clip(cores, 1, max_cores))
    else:
        return max_cores


def relative_cores(jobs):
    """
    Return the actual number of cores to use for a given number of jobs.

    Returns 1 in cases where jobs is None or 0.  If jobs is less than zero,
    the returned value will be max_available_cores + jobs + 1.  i.e., -1 will
    use all available cores.

    Parameters
    ----------
    jobs : int or float or None

    Returns
    -------
    n_cores : int
        The number of cores to use which will always be in the range 1 ->
        max_available_cores.
    """
    if jobs is None:
        return 1

    jobs = float(jobs)
    if jobs == 0:
        return 1

    max_cores = get_core_number()

    if jobs % 1 != 0:
        jobs *= max_cores

    jobs = int(jobs)
    if jobs < 0:
        cores = max_cores + jobs + 1
    else:
        cores = jobs
    return get_core_number(cores=cores)


def valid_relative_jobs(jobs):
    """
    Return a valid number of jobs in the range 1 <= jobs <= max_cores.

    Parameters
    ----------
    jobs : int
        An positive or negative integer.  Negative values are processed as
        max_cores - jobs + 1.

    Returns
    -------
    valid_jobs : int
        The number of jobs available to process.
    """
    return get_core_number(relative_cores(jobs))


def multitask(func, iterable, args, kwargs, jobs=None, skip=None,
              max_nbytes='1M', force_threading=False, force_processes=False,
              logger=None):
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
    max_nbytes : int or str or None, optional
        Threshold on the size of arrays passed to the workers that triggers
        automated memory mapping in temp_folder. Can be an int in Bytes, or a
        human-readable string, e.g., ‘1M’ for 1 megabyte. Use None to disable
        memmapping of large arrays. Only active when backend=”loky” or
        “multiprocessing”.  The default is currently set to '1M' for
        consistency with the joblib library.  Note that memmapping disallows
        in-place modification of data, so if this functionality is required set
        `max_nbytes` to `None`.
    force_threading : bool, optional
        If `True`, force joblib to run parrallel jobs using threads so that
        shared memory can be used.  Otherwise, threading will only occur
        when parallel processing is spawned from a child process of the main
        thread.
    force_processes : bool, optional
        If `True`, force joblib to run parrallel jobs using CPUs rather than
        threads.  This can sometimes lead to unexpected outcomes when the
        multiprocessing is launched from a non-main thread.  Pickling arguments
        prior and return values during processing is recommended in this case.
    logger : logging.Logger, optional
        If supplied, will attempt to produce sensibly ordered logs for the
        multiprocessing tasks for all handlers.

    Returns
    -------
    result : list
        The final output where result[i] = func(args, iterable[i]).  Will be
        of length len(`iterable`) if `skip` is None, otherwise
        len(iterable) - sum(skip).
    """
    if jobs in [None, 0, 1]:
        return _serial(func, args, kwargs, iterable, skip=skip)

    return _parallel(
        int(jobs), func, args, kwargs, iterable, skip=skip,
        max_nbytes=max_nbytes, force_threading=force_threading,
        force_processes=force_processes, logger=logger)


def _wrap_function(func, args, kwargs):
    """
    Wrap given arguments and keyword arguments to a function.

    Removes The requirement of supplying args and kwargs to the given function.
    :func:`multitask` should be run on a function supplied in a very strict
    format, and actually only takes one single runtime argument.  Functions
    should be designed so that they are of the form:

    def some_function(args, kwargs, run_time_argument):
        <code here>

    or

    def some_function(args, run_time_argument):
        <code here>

    All calls to `some_function` will always contain the same arguments and
    keyword arguments, but a different run time argument will be supplied to
    each call.  An easy way to set this up is to set args and kwargs to a list
    of arguments for each call, and then select which set to use using the
    run time argument.  For example, you could set up something like:

    >>> arguments = [(x, x + 2) for x in range(10)]
    >>> keyword_arguments = {'multiply': True}
    >>> def my_func(my_args, my_kwargs, index):
    ...     x, y = my_args[index]
    ...     if my_kwargs['multiply']:
    ...         return x * y
    ...     else:
    ...         return x + y

    :func:`_wrap_function` will remove the requirement to always specify args
    and kwargs for each call, so `my_func(args, kwargs, index)` is the same
    as `_wrap_function(my_func, args, kwargs)(index)`.

    Parameters
    ----------
    func : function
        The function to wrap.
    args : tuple
        The function arguments.
    kwargs : dict or None
        Any function keyword arguments.

    Returns
    -------
    wrapped_function : function
    """
    if kwargs is None:
        return partial(func, args)
    else:
        return partial(func, args, kwargs)


def _wrap_function_with_logger(func, args, kwargs, logger_pickle_file,
                               log_directory):
    """
    Wrap a function and also halt all logging until the process is complete.

    Please see :func:`_wrap_function` for details on how the function should
    be implemented.  Additionally, a logger is unpickled and used to emit any
    log messages one the function is complete.

    Parameters
    ----------
    func : function
        The function to wrap.
    args : tuple
        The function arguments.
    kwargs : dict or None
        Any function keyword arguments.
    logger_pickle_file : str
        A filename containing the logger to unpickle and use.
    log_directory : str
        Pickles the log records to the given directory.

    Returns
    -------
    wrapped_function : function
    """
    wrapped = _wrap_function(func, args, kwargs)
    return partial(wrapped_with_logger, wrapped, logger_pickle_file,
                   log_directory)


def wrapped_with_logger(func, logger_pickle_file, log_directory,
                        run_arg_and_identifier):
    """
    Return the results of the function in multitask and save log records.

    Parameters
    ----------
    func : function
        The function to wrap.
    logger_pickle_file : str
        The file path to the pickled logger (:class:`logging.Logger`).
    log_directory : str
        The directory in which to store the log records for each run.
    run_arg_and_identifier : 2-tuple
        Any run time argument that the wrapped function returned by
        :func:`_wrap_function` requires (object), and an integer identifier
        signifying it's position in a list of run arguments.

    Returns
    -------
    results : object
        The results of running `func` on a given run time argument.
    """
    logger, _ = unpickle_file(logger_pickle_file)
    run_arg, identifier = run_arg_and_identifier
    log_basename = f"multitask_log_{identifier}.p"
    record_file = os.path.join(log_directory, log_basename)
    with log_records_to_pickle_file(logger, record_file):
        result = func(run_arg)
    return result


def wrap_function(func, args, kwargs=None, logger=None, log_directory=None):
    """
    Wrap a function for use with :func:`multitask`.

    Parameters
    ----------
    func : function
        The function to wrap.
    args : tuple
        The function arguments.
    kwargs : dict, optional
        Any function keyword arguments.
    logger : logging.Logger or str, optional
        A logger used to output any log messages once complete.  If supplied,
        a valid `log_directory` must also be supplied.  A path to a pickle file
        containing the logger may also be supplied.
    log_directory : str, optional
        If supplied together with a `logger`, will store all log records to
        a pickle file in the given directory.

    Returns
    -------
    wrapped_function, log_pickle_file : function, str
        The wrapped function and the file location of the pickle file for any
        supplied logger.  If no logger was supplied, this value will be `None`.
    """
    if logger is None:
        return _wrap_function(func, args, kwargs), None

    if not isinstance(log_directory, str) or not os.path.isdir(
            log_directory):
        raise ValueError(f"Must supply a valid log directory.  Received "
                         f"{log_directory}.")

    if isinstance(logger, str) and os.path.isfile(logger):
        pickle_file = logger
    else:
        logger_id = id((logger, args))
        tmp_fh, tmp_fname = tempfile.mkstemp(
                prefix=f'multitask_logger_{logger_id}', suffix='.p')
        os.close(tmp_fh)
        pickle_file = pickle_object(logger, tmp_fname)
    multi_func = _wrap_function_with_logger(
        func, args, kwargs, pickle_file, log_directory)
    return multi_func, pickle_file


def _serial(func, args, kwargs, iterable, skip=None):
    """
    Processes tasks serially without any multiprocessing.

    Parameters
    ----------
    func : function
        The function to repeat multiple times on different sets of arguments.
        Must be of the form func(args, i) if `kwargs` is None, or
        func(args, kwargs, i) otherwise, where `i` is a member of `iterable`.
    args
        Anything that should be passed to `func` as the first argument.  The
        intended use is such that the output of `func` is
        result[i] = f(args[iterable[i]]).
    kwargs : None or anything
        If set to None, multitask assumes `func` is of the form func(args, i).
        Otherwise, multitask assumes `func` is of the form
        func(args, kwargs, i).
    iterable : iterable
        A Python object that can be iterated though such that each member
        can be passed to `func` as the final argument.
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
    multi_func, _ = wrap_function(func, args, kwargs=kwargs)
    result = []

    if skip is None:
        for thing in iterable:
            result.append(multi_func(thing))
    else:
        for (skip_it, thing) in zip(skip, iterable):
            if skip_it:
                continue
            else:
                result.append(multi_func(+thing))

    return result


def _parallel(jobs, func, args, kwargs, iterable, skip=None,
              force_threading=False, force_processes=False, package=None,
              logger=None, **joblib_kwargs):
    """

    Process a given list of jobs in parallel.

    Joblib and the multiprocessing package are currently used to process jobs
    in parallel.  Generally, the joblib package appears to produce results
    faster than the base Python multiprocessing package but cannot always be
    implemented.  The default package will generally be set to joblib unless
    multiprocessing has been specifically requested but we :func:`_parallel`
    has been started on a non-main thread.

    Unfortunately, the joblib loky backend will always start new processes
    using a fork followed by an exec() which is not safe for certain operating
    systems, especially MacOS.  For Python versions > 3.8, the multiprocessing
    module starts new processes using the spawn method which is safe for nearly
    all platforms, but does introduce additional overhead.  In cases where
    multiprocessing is required and the Python version is < 3.8, only serial
    reduction will be possible.

    Parameters
    ----------
    jobs : int
        The number of jobs to perform in parallel.
    func : function
        The function to repeat multiple times on different sets of arguments.
        Must be of the form func(args, i) if `kwargs` is None, or
        func(args, kwargs, i) otherwise, where `i` is a member of `iterable`.
    args
        Anything that should be passed to `func` as the first argument.  The
        intended use is such that the output of `func` is
        result[i] = f(args[iterable[i]]).
    kwargs : None or anything
        If set to None, multitask assumes `func` is of the form func(args, i).
        Otherwise, multitask assumes `func` is of the form
        func(args, kwargs, i).
    iterable : iterable
        A Python object that can be iterated though such that each member
        can be passed to `func` as the final argument.
    skip : array_like of bool, optional
        Should be of len(`iterable`), where a value of True signifies that
        processing should be skipped for that task and omitted from the
        final result.
    force_threading : bool, optional
        If `True`, run the jobs in parallel using a thread pool.  This is
        preferable for high I/O and low CPU intensive operations or when access
        to shared variables is required.  Cannot be used in conjunction with
        `force_processes`.
    force_processes : bool, optional
        If `True`, run the jobs in parallel using multiprocessing.  This is
        preferable for computationally expensive tasks but creates copies of
        the main process on each CPU, so no ready access to shared variables
        exists and startup costs are greater.
    package : str, optional
        The multiprocessing package to use.  May be one of {'joblib',
        'multiprocessing', None}.  `None` (the default) will estimate the best
        package at runtime.
    logger : Logger, optional
        The logger with which to emit any messages during `func`.
    joblib_kwargs : dict, optional
        Optional keyword arguments to pass into :class:`joblib.Parallel` if
        applicable.  The `require` and `backend` options will be overwritten
        if threading is used.

    Returns
    -------
    result : list
        The final output where result[i] = func(args, iterable[i]).  Will be
        of length len(`iterable`) if `skip` is None, otherwise
        len(iterable) - sum(skip).
    """
    if force_threading and force_processes:
        raise ValueError("Can either force threading or processes, not both.")

    # Check if this is just a single job.
    if jobs is None or jobs < 2:
        return _serial(func, args, kwargs, iterable, skip=skip)
    if skip is None:
        run_args = list(iterable)
    else:
        run_args = [x[1] for x in zip(skip, iterable) if not x[0]]

    required_jobs = int(np.clip(jobs, 1, len(run_args)))
    if required_jobs == 1:
        return _serial(func, args, kwargs, iterable, skip=skip)

    # Determine which package to use.
    if not in_windows_os():
        reason = 'not installed'
        try:
            from joblib import delayed, Parallel
            have_joblib = True
        except ImportError:  # pragma: no cover
            have_joblib = False
            delayed = Parallel = None
    else:
        reason = 'not available on Windows'
        have_joblib = False
        delayed = Parallel = None

    requested_package = package
    if package == 'joblib' and not have_joblib:  # pragma: no cover
        raise ValueError(f"Cannot use joblib package: {reason}.")
    elif package is None:
        package = 'joblib' if have_joblib else 'multiprocessing'
    elif package not in ['multiprocessing', 'joblib']:
        raise NotImplementedError(f"The {package} package is not supported.")

    if force_processes:
        use_threads = False
        if not in_main_thread():  # pragma: no cover
            if sys.version_info < (3, 8, 0):
                log.warning("Multiprocessing is not available from a child "
                            "thread for Python versions < 3.8.0: will process "
                            "serially")
                return _serial(func, args, kwargs, iterable, skip=skip)

            if package == 'joblib':
                package = 'multiprocessing'
                if requested_package == 'joblib':
                    log.warning("Cannot use joblib for multiprocessing from "
                                "child thread: will use the multiprocessing "
                                "package.")

    elif force_threading:
        use_threads = True
    else:
        use_threads = not in_main_thread()

    if logger is not None:
        log_directory = tempfile.mkdtemp(prefix='multitask_temp_log_dir_')
        run_args = [(x, i) for (i, x) in enumerate(run_args)]
        initial_log_level = logger.level
    else:
        log_directory = None
        initial_log_level = None

    multi_func, log_pickle_file = wrap_function(
        func, args, kwargs=kwargs, logger=logger, log_directory=log_directory)

    if package == 'multiprocessing':
        pool_class = mp.pool.ThreadPool if use_threads else mp.Pool
        with pool_class(processes=required_jobs) as pool:
            result = pool.map(multi_func, run_args)
            pool.close()
            pool.join()

        purge_multitask_logs(log_directory, log_pickle_file, use_logger=logger)
        return result

    # Joblib processing...
    joblib_kwargs['n_jobs'] = required_jobs
    if use_threads:
        joblib_kwargs['require'] = 'sharedmem'
        joblib_kwargs['backend'] = 'threading'

    if 'mmap_mode' not in joblib_kwargs:
        joblib_kwargs['mmap_mode'] = 'r'  # was previously r

    result = [delayed(multi_func)(run_arg) for run_arg in run_args]

    # joblib does not reliably close child processes
    # Store current child processes
    current_process = psutil.Process()
    subprocesses_before = set(
        [p.pid for p in current_process.children(recursive=True)])

    executor = Parallel(**joblib_kwargs)
    processed_result = executor(result)
    if logger is not None:
        logger.setLevel(initial_log_level)

    purge_multitask_logs(log_directory, log_pickle_file, use_logger=logger)

    # Terminate new child processes that are still running.
    subprocesses_after = set(
        [p.pid for p in current_process.children(recursive=True)])
    terminate = (subprocesses_after - subprocesses_before)

    if in_windows_os():
        for subprocess in terminate:
            try:
                os.kill(subprocess, signal.CTRL_BREAK_EVENT)
            except ProcessLookupError:
                pass
    else:
        for subprocess in terminate:
            try:
                os.killpg(subprocess, signal.SIGTERM)
            except ProcessLookupError:
                pass

    return processed_result


def pickle_object(obj, filename):
    """
    Pickle a object and save to the given filename.

    Parameters
    ----------
    obj : object
        The object to pickle.
    filename : str or None
        If `filename` points to a writeable on-disk location, `obj` will
        be pickled and saved to that location.  If `None`, nothing will
        happen.

    Returns
    -------
    output : str or object
        Either the `filename` if the object was pickled, or `obj` if it
        wasn't.
    """
    if filename is None:
        return obj
    with open(filename, 'wb') as f:
        cloudpickle.dump(obj, f)
    return filename


def unpickle_file(filename):
    """
    Unpickle a string argument if it is a file, and return the result.

    Parameters
    ----------
    filename : object or str
        If the argument is a string and a valid file path, it will be
        unpickled and the result will be available in the result.

    Returns
    -------
    obj, pickle_file : object, str
        If the argument passed in was not a string or an invalid file,
        the resulting output `obj` will be `argument` and `pickle_file` will
        be `None`.  If `argument` was a valid file path to a pickle file,
        `obj` will be the unpickled result, and `pickle`file` will be
        `argument`.
    """
    if not isinstance(filename, str):
        return filename, None

    if not os.path.isfile(filename):
        log.warning(f"Pickle file not found: {filename}")
        return filename, None
    pickle_file = filename
    with open(pickle_file, 'rb') as f:
        obj = cloudpickle.load(f)
    return obj, pickle_file


def pickle_list(object_list, prefix=None, naming_attribute=None,
                class_type=None):
    """
    Pickle a list of objects to a temporary directory.

    The list will be updated in-place, with each element being replaced
    by the on-disk file path to the pickle file in which it is saved.

    Parameters
    ----------
    object_list : list (object)
        A list of things to pickle.
    prefix : str, optional
        The prefix for the temporary directory in which to store the pickle
        files.  See :func:`tempfile.mkdtemp` for further information.
    naming_attribute : str, optional
        The attribute used to name the pickle file.  If not supplied,
        defaults to id(object).
    class_type : class, optional
        If supplied, only objects of this class type will be pickled.

    Returns
    -------
    temporary_directory : str
        The temporary directory in which the objects are saved as pickle
        files.
    """
    directory = tempfile.mkdtemp(prefix=prefix)
    for i, obj in enumerate(object_list):
        if class_type is not None:
            if not isinstance(obj, class_type):
                continue
        if naming_attribute is not None:
            filename = f'{getattr(obj, naming_attribute, id(obj))}.p'
        else:
            filename = f'{id(obj)}.p'
        pickle_file = os.path.join(directory, filename)
        with open(pickle_file, 'wb') as f:
            cloudpickle.dump(obj, f)
        object_list[i] = pickle_file
    return directory


def unpickle_list(pickle_files, delete=True):
    """
    Restore pickle files to objects in-place.

    Parameters
    ----------
    pickle_files : list (str)
        A list of on-disk pickle files to restore.  The restored objects
        will replace the filepath for each element in the list.
    delete : bool, optional
        If `True`, delete each pickle file once it has been restored.

    Returns
    -------
    None
    """
    if pickle_files is None:
        return
    result = pickle_files
    for i, pickle_file in enumerate(pickle_files):
        obj, filename = unpickle_file(pickle_file)
        if filename is None:  # not a valid file
            continue
        result[i] = obj
        if delete:
            os.remove(filename)


def in_main_thread():
    """
    Return whether the process is running in the main thread.

    Returns
    -------
    main_thread: bool
        `True` if this process is running in the main thread, and `False` if
        it is running in a child process.
    """
    return threading.current_thread() == threading.main_thread()


@contextmanager
def log_with_multi_handler(logger):
    """
    Context manager to temporarily log messages for unique processes/threads

    Temporarily disables all log handlers and outputs the results to a
    dictionary of the form {(process, thread): list(records)} where process
    is returned by :func:`multiprocessing.current_process()` and thread is
    returned by :func:`threading.current_thread()`.

    Parameters
    ----------
    logger : logging.Logger

    Yields
    ------
    multi_handler : MultitaskHandler
    """
    original_handlers = logger.handlers.copy()
    multi_handler = MultitaskHandler()
    logger.addHandler(multi_handler)

    initial_level = logger.level
    logger.setLevel('DEBUG')  # Need to capture all records

    for handler in original_handlers:
        logger.removeHandler(handler)

    yield multi_handler
    logger.setLevel(initial_level)

    for handler in original_handlers:
        logger.addHandler(handler)
    logger.removeHandler(multi_handler)


@contextmanager
def log_for_multitask(logger):
    """
    Context manager to output log messages during multiprocessing.

    Stores all log messages during multiprocessing, and emits them using
    the given logger once complete.

    Parameters
    ----------
    logger : logging.Logger

    Yields
    ------
    None
    """
    initial_level = logger.level
    with log_with_multi_handler(logger) as multi_handler:
        yield multi_handler

    logger.setLevel(initial_level)

    multi_handler.reorder_records()
    handlers = []
    for handler in logger.handlers:
        if isinstance(handler, MultitaskHandler):
            for record in multi_handler.records:
                if record not in handler.records:
                    handler.records.append(record)
        else:
            handlers.append(handler)

    if len(handlers) == 0:
        return

    for record in multi_handler.records:
        logger.handle(record[-1])


def purge_multitask_logs(log_directory, log_pickle_file, use_logger=None):
    """
    Remove all temporary logging files/directories and handle log records.

    The user must supply a `log_directory` containing pickle files of the log
    records to handle.  The `log_pickle_file` contains the logger used to
    handle these records.  Following completion, both the 'log_directory` and
    `log_pickle_file` will be deleted from the file system.

    Parameters
    ----------
    log_directory : str
        The directory in which the log records for each run were stored.  This
        directory will be removed.
    log_pickle_file : str
        The pickle file containing the logger for multitask.  This will be
        removed.
    use_logger : Logger, optional
        The logger to handle any log records.  If not supplied, defaults to
        that found in the log_pickle_file.

    Returns
    -------
    None
    """
    if not isinstance(log_pickle_file, str) or not os.path.isfile(
            log_pickle_file):
        return

    if not isinstance(log_directory, str) or not os.path.isdir(log_directory):
        os.remove(log_pickle_file)
        return

    if not isinstance(use_logger, logging.Logger):
        with open(log_pickle_file, 'rb') as f:
            logger = cloudpickle.load(f)
    else:
        logger = use_logger

    pickle_files = os.listdir(log_directory)
    if logger is None or len(pickle_files) == 0:
        os.remove(log_pickle_file)
        shutil.rmtree(log_directory)
        return

    id_search = regex.compile(r'multitask\_log\_(.*)\.p$')
    identifiers = {}
    for pickle_file in pickle_files:
        search = id_search.search(pickle_file)
        if search is None:
            continue
        identifiers[int(search.group(1))] = pickle_file

    job_logs = [os.path.join(log_directory, identifiers[i])
                for i in sorted(identifiers.keys())]

    unpickle_list(job_logs)  # deletes files and converts to log records

    for job_log in job_logs:
        for record in job_log:
            if logger.isEnabledFor(record.levelno):
                logger.handle(record)

    os.remove(log_pickle_file)
    shutil.rmtree(log_directory)


@contextmanager
def log_records_to_pickle_file(logger, pickle_file):
    """
    Store the log records in a pickle file rather than emitting.

    Parameters
    ----------
    logger : logging.Logger
    pickle_file : str
        The path to the pickle file that will contain the log records.

    Yields
    ------
    None
    """
    initial_level = logger.level
    with log_with_multi_handler(logger) as multi_handler:
        yield

    logger.setLevel(initial_level)
    multi_handler.reorder_records()
    standard_records = [x[-1] for x in multi_handler.records]
    with open(pickle_file, 'wb') as f:
        cloudpickle.dump(standard_records, f)


def in_windows_os():
    """
    Return `True` if running from a Windows OS.

    Returns
    -------
    bool
    """
    return os.name.lower().strip() == 'nt'


class MultitaskHandler(logging.Handler):
    """A log handler for multitask."""

    def __init__(self):
        """
        Initialize a multitask log Handler.

        The multitask log handler is designed to separately store log messages
        for each process or thread in order to retrieve those messages later
        for standard logging.  This allows log messages to be output in order
        for each process.
        """
        logging.Handler.__init__(self)
        self.records = []

    def emit(self, record):
        """
        Emit a log record.

        Stores the record in the lookup dictionary for the
        given process/thread. Each message is stored in the received
        order for later retrieval once whatever multiprocessing
        job is complete.

        Parameters
        ----------
        record : logging.LogRecord
            The record to emit.

        Returns
        -------
        None
        """
        process = mp.current_process()
        thread = threading.current_thread()
        emit_time = time.time()
        self.records.append((emit_time, process, thread, record))

    def reorder_records(self):
        """
        Re-order the records in a sensible order.

        The records are sorted by process and then thread in chronological
        order.  I.e., records are grouped by processes, starting with the
        first process that appears in the logs and then within that process
        group, a similar grouping is performed for each child thread.  Each
        process-thread grouping will contain a list of log records in the
        order that they where emitted.

        Note that each record is a tuple of the form (time, process, thread,
        log_record).

        Returns
        -------
        None
        """
        if self.records is None or len(self.records) == 0:
            return

        # Try to figure out the main and child threads which can be very
        # hard since there is no record in the thread objects, and GUIs may
        # not always launch processes from the main thread.

        # Need to separate out logs for separate processes
        process_logs = {}

        for index, info in enumerate(self.records):
            t, process, thread, record = info

            if process not in process_logs:
                process_logs[process] = []

            process_log = process_logs[process]
            if info not in process_log:
                process_log.append(info)

        ordered = []
        for process, process_info in process_logs.items():
            ordered_logs = []
            main_thread = None
            thread_logs = {}
            for info in process_info:
                t, _, thread, record = info
                if main_thread is None:
                    main_thread = thread
                # Note that coverage cannot be determined in threads.
                if thread == main_thread:
                    if len(thread_logs) > 0:  # pragma: no cover
                        for thread_info in thread_logs.values():
                            ordered_logs.extend(thread_info)
                        thread_logs = {}

                    ordered_logs.append(info)
                    continue

                if thread not in thread_logs:  # pragma: no cover
                    thread_logs[thread] = []
                thread_log = thread_logs[thread]  # pragma: no cover
                thread_log.append(info)  # pragma: no cover

                if (thread.name == 'MainThread'
                        or not thread.daemon):   # pragma: no cover
                    if len(thread_logs) > 0:
                        for thread_info in thread_logs.values():
                            ordered_logs.extend(thread_info)
                        thread_logs = {}
                    main_thread = thread

            if len(thread_logs) > 0:  # pragma: no cover
                for thread_info in thread_logs.values():
                    ordered_logs.extend(thread_info)

            ordered.extend(ordered_logs)

        self.records = ordered
