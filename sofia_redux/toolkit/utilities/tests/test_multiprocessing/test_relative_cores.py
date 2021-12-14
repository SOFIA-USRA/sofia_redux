# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import relative_cores

import psutil


def test_relative_cores():
    max_cores = psutil.cpu_count()
    assert relative_cores(None) == 1
    assert relative_cores(0) == 1
    assert relative_cores(0.5) == max(int(max_cores * 0.5), 1)
    assert relative_cores(-1) == max_cores
