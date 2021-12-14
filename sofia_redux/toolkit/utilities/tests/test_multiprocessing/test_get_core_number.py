# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import get_core_number


def test_get_core_number():
    maxcores = get_core_number()
    assert get_core_number(cores=False) == 1
    assert get_core_number(cores=1e6) == maxcores
    assert get_core_number(cores=-1) == 1
    assert get_core_number(cores='a') == maxcores
