# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import in_main_thread

import multiprocessing as mp
import threading
import numpy as np


def test_in_main_thread():
    assert in_main_thread()


def test_not_in_main_thread():

    def some_func(x):
        print(f"x = {x}; process = {mp.current_process()}; "
              f"thread = {threading.current_thread()}")
        return in_main_thread()

    with mp.pool.ThreadPool(processes=2) as pool:
        results = pool.map(some_func, range(2))
        pool.close()
        pool.join()

    assert len(results) == 2 and np.allclose(results, False)
