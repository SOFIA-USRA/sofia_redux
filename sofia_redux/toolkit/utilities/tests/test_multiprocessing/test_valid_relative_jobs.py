# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import valid_relative_jobs

import psutil


def test_relative_cores():
    max_cores = psutil.cpu_count()
    assert valid_relative_jobs(None) == 1
    assert valid_relative_jobs(0) == 1
    assert valid_relative_jobs(0.75) == max(int(max_cores * 0.75), 1)
    assert valid_relative_jobs(-2) == max(max_cores - 1, 1)
    assert valid_relative_jobs(-max_cores - 1) == 1
