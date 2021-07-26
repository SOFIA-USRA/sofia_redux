import numpy as np

from sofia_redux.toolkit.utilities.func import nansum


def test_1d():
    a = np.arange(10, dtype=float)
    assert nansum(a) == 45
    a[4:8] = np.nan
    assert nansum(a) == 23
    a.fill(np.nan)
    assert np.isnan(nansum(a))


def test_2d():
    a = np.arange(25, dtype=float).reshape((5, 5))
    assert nansum(a) == 300
    assert np.allclose(nansum(a, axis=0), [50, 55, 60, 65, 70])
    assert np.allclose(nansum(a, axis=1), [10, 35, 60, 85, 110])
    a[2:4, 1:3] = np.nan
    assert np.allclose(nansum(a, axis=0), [50, 28, 31, 65, 70])
    assert np.allclose(nansum(a, axis=1), [10, 35, 37, 52, 110])
    a[2] = np.nan
    assert np.allclose(nansum(a, axis=1), [10, 35, np.nan, 52, 110],
                       equal_nan=True)
    a.fill(np.nan)
    assert np.isnan(nansum(a))
    assert np.allclose(nansum(a, axis=0), [np.nan] * 5, equal_nan=True)
    assert np.allclose(nansum(a, axis=1), [np.nan] * 5, equal_nan=True)


def test_missing():
    a = np.full(5, np.nan)
    result = nansum(a, missing=0)
    assert result == 0
    assert isinstance(result, np.float)  # Check casting ok


def test_non_array():
    # non-arrays should also be directly handled
    a = [1, 2, 3]
    assert nansum(a) == 6

    a = [1, 2, 3, np.nan]
    assert nansum(a) == 6

    a = [np.nan, np.nan, np.nan]
    assert np.isnan(nansum(a))
