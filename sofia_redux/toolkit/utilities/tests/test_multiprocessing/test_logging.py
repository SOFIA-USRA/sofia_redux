# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import (
    log_with_multi_handler, log_for_multitask, MultitaskHandler,
    log_records_to_pickle_file, unpickle_file, pickle_object,
    purge_multitask_logs, wrapped_with_logger)

from sofia_redux.toolkit.utilities.tests.test_multiprocessing.multi_functions \
    import MultiLogger

from astropy import log
import os
import psutil
import pytest


def records_valid(records):
    """Check the records for the multitask handler are ok"""
    # Processes should always be in order
    # Threads are in the order they were processed
    if isinstance(records[0], tuple):
        log_values = [int(x[-1].msg.split(' ')[0]) for x in records]
    else:
        log_values = [int(x.msg.split(' ')[0]) for x in records]
    assert log_values[0] == 1000
    assert sorted(log_values[1:3]) == [1001, 1002]
    assert log_values[3] == 1000
    assert sorted(log_values[4:6]) == [1011, 1012]
    assert log_values[6] == 2000
    assert sorted(log_values[7:9]) == [2001, 2002]
    assert log_values[9] == 2000
    assert sorted(log_values[10:12]) == [2011, 2012]
    return True


def test_wrapped_with_logger(tmpdir):
    log_directory = str(tmpdir.mkdir('test_logging'))
    log_pickle_file = os.path.join(log_directory, 'logger.p')
    log_pickle_file = pickle_object(log, log_pickle_file)
    run_arg_and_identifier = (1, 5)

    def my_func(x):
        log.info(f"TEST received {x}")
        return x

    wrapped_result = wrapped_with_logger(my_func, log_pickle_file,
                                         log_directory, run_arg_and_identifier)
    assert wrapped_result == 1
    assert 'multitask_log_5.p' in os.listdir(log_directory)


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_multitask_handler():
    handler = MultitaskHandler()
    log.addHandler(handler)

    handler.reorder_records()
    assert handler.records == []

    log.info("This is the main process")

    test_obj = MultiLogger(logger=log)
    test_obj.process_func()
    log.removeHandler(handler)
    handler.reorder_records()

    assert records_valid(handler.records[1:])


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_log_with_multi_handler():
    original_handlers = log.handlers
    test_obj = MultiLogger(logger=log)
    with log_with_multi_handler(log) as handler:
        test_obj.process_func()
    assert records_valid(handler.records)
    assert log.handlers == original_handlers


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_log_for_multitask():

    original_handlers = log.handlers.copy()
    test_obj = MultiLogger(logger=log)
    with log_for_multitask(log) as handler:
        test_obj.process_func()
    assert records_valid(handler.records)
    assert log.handlers == original_handlers

    new_handler = MultitaskHandler()
    handler.records = []
    log.addHandler(new_handler)

    with log_for_multitask(log) as handler:
        test_obj.process_func()
    assert records_valid(handler.records)
    new_handler.reorder_records()
    assert records_valid(new_handler.records)

    log.removeHandler(new_handler)
    for handler in original_handlers:
        log.removeHandler(handler)

    with log_for_multitask(log) as handler:
        test_obj.process_func()
    assert records_valid(handler.records)

    for handler in original_handlers:  # pragma: no cover
        log.addHandler(handler)


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_log_records_to_pickle_file(tmpdir):
    test_dir = tmpdir.mkdir('test_multilog')
    test_file = str(test_dir.join('logfile.p'))
    test_obj = MultiLogger(logger=log)
    with log_records_to_pickle_file(log, test_file):
        test_obj.process_func()
    records, _ = unpickle_file(test_file)
    assert records_valid(records)


@pytest.mark.skipif(psutil.cpu_count() < 2, reason='Require multiple CPUs')
def test_purge_multitask_logs(tmpdir):
    log_directory = str(tmpdir.mkdir('test_multilog'))
    n_logs = 2
    log_files = [os.path.join(log_directory, f'multitask_log_{x}.p')
                 for x in range(n_logs)]
    log_pickle_file = pickle_object(
        log, os.path.join(log_directory, 'logger.p'))

    test_obj = MultiLogger(logger=log)

    for i in range(n_logs):
        with log_records_to_pickle_file(log, log_files[i]):
            test_obj.process_func()

    delete_logger = pickle_object(
        log, os.path.join(log_directory, 'delete_me.p'))
    assert os.path.isfile(delete_logger)

    purge_multitask_logs(None, None)

    purge_multitask_logs(None, delete_logger)
    assert not os.path.isfile(delete_logger)

    empty_directory = str(tmpdir.mkdir('empty_directory'))
    delete_logger = pickle_object(
        log, os.path.join(log_directory, 'delete_me.p'))
    assert os.path.isdir(empty_directory)
    purge_multitask_logs(empty_directory, delete_logger)
    assert not os.path.isfile(delete_logger)
    assert not os.path.isdir(empty_directory)

    test_directory = str(tmpdir.mkdir('empty_directory'))
    delete_logger = pickle_object(
        log, os.path.join(log_directory, 'delete_me.p'))
    bad_log = os.path.join(empty_directory, 'bad_log.p')
    with open(bad_log, 'w') as f:
        f.write('foo')
    assert os.path.isfile(bad_log)
    purge_multitask_logs(test_directory, delete_logger)
    assert not os.path.isfile(delete_logger)
    assert not os.path.isdir(test_directory)

    purge_multitask_logs(log_directory, log_pickle_file)
    assert not os.path.isfile(log_pickle_file)
    assert not os.path.isdir(log_directory)
