# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.stats.stats import find_outliers


def test_expected_output():
    rand = np.random.RandomState(42)
    data = rand.normal(loc=10.0, scale=2.0, size=(100, 100))
    result = find_outliers(data, threshold=100)
    assert result.all()
    result = find_outliers(data, threshold=1)
    assert not result.all()
    assert result.any()
    data[0, 0] = 2 * 200
    result = find_outliers(data, threshold=100)
    assert not result[0, 0]
    assert np.sum(~result) == 1

    data = rand.rand(10)
    data[5] = np.nan
    result = find_outliers(data, threshold=100, keepnans=False)
    assert not result[5]
    result = find_outliers(data, threshold=100, keepnans=True)
    assert result[5]


def test_axis():
    rand = np.random.RandomState(42)
    data = rand.normal(loc=10.0, scale=2.0, size=(100, 100))
    for i, row in enumerate(data):
        row *= i + 1
        row[50] *= 5

    # In this case, if we do not check for outliers relative to
    # each row, then the outliers in the first few rows will be
    # missed if we base rejection on the stats of the full set.
    flag0 = ~find_outliers(data, threshold=5)
    flag1 = ~find_outliers(data, threshold=5, axis=0)
    nflag0 = np.sum(flag0, axis=0)
    nflag1 = np.sum(flag1, axis=0)
    assert nflag1[50] > nflag0[50]


def test_all_nans():
    data = np.full(100, np.nan)
    result = find_outliers(data, threshold=100, keepnans=False)
    assert not result.any()
    result = find_outliers(data, threshold=100, keepnans=True)
    assert result.all()
