# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.utilities.multiprocessing import multitask

import numpy as np
import pytest


def adder(xy, i):
    return xy[0, i] + xy[1, i]


@pytest.fixture
def adder_data():
    n = 100
    xy = np.arange(n)
    xy = np.vstack((xy, xy + 100))
    expected = np.sum(xy, axis=0)
    iterable = list(range(n))
    return xy, iterable, expected


def test_multitask(adder_data):
    xy, iterable, expected = adder_data
    args = xy
    kwargs = None

    for jobs in [None, 0, 1, 2]:
        result = multitask(adder, iterable, args, kwargs, jobs=jobs)
        assert np.allclose(result, expected)

    skip = np.full(len(iterable), False)
    skip[0] = True
    result = multitask(adder, iterable, args, kwargs, jobs=2, skip=skip)
    assert np.allclose(result, expected[~skip])
    result = multitask(adder, iterable, args, kwargs, jobs=1, skip=skip)
    assert np.allclose(result, expected[~skip])
