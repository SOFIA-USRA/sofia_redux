# Licensed under a 3-clause BSD style license - see LICENSE.rst
import threading

from astropy import log
from astropy.time import Time
import multiprocessing as mp
import numpy as np
import time

from sofia_redux.toolkit.utilities.multiprocessing import multitask


class MultiLogger(object):
    """
    Class to check logging from multiprocessing operations.
    """
    def __init__(self, logger=None):
        if logger is None:
            self.logger = log
        else:
            self.logger = logger

    @classmethod
    def spew_messages(cls, args, run_arg):
        """
        Create a log message.
        """
        x = args[run_arg]
        process = mp.current_process()
        thread = threading.current_thread()
        log.info(f"{x} Info at {Time.now()} {process} {thread}")
        # log.warning(f"{x} Warning at {Time.now()}")
        time.sleep(np.random.random(1)[0] / 5)
        return x + 100

    @classmethod
    def thread_func(cls, args, run_arg):  # pragma: no cover
        """
        A function to exercise thread processes.
        """
        x = args[run_arg]
        processes = 2
        cls.spew_messages(args, run_arg)
        multi_args = [x + i + 1 for i in range(processes)]
        results = multitask(cls.spew_messages, range(processes), multi_args,
                            None, force_threading=True, jobs=processes,
                            logger=log)

        multi_args_2 = [x + i + 11 for i in range(processes)]
        cls.spew_messages(args, run_arg)
        _ = multitask(
            cls.spew_messages, range(processes), multi_args_2,
            None, force_threading=True, jobs=processes,
            logger=log)

        return results

    def process_func(self, logger=None):  # pragma: no cover
        """
        A function to exercise processes.
        """
        processes = 2
        if logger is None:
            logger = self.logger
        multi_args = [(x + 1) * 1000 for x in range(processes)]

        results = multitask(self.thread_func, range(processes), multi_args,
                            None, force_processes=True, jobs=processes,
                            logger=logger)
        return results
