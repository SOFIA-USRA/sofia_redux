# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.flags.flagged_array import FlaggedArray
from sofia_redux.scan.utilities.range import Range


@pytest.fixture
def dummy_array():
    """
    Return a test FlaggedArray.

    Returns
    -------
    FlaggedArray
    """
    data = np.arange(7 * 8).reshape(7, 8).astype(float)
    array = FlaggedArray(data)
    return array


@pytest.fixture
def smooth_convolve():
    smooth = np.asarray(
        [[4.5, 5., 6., 7., 8., 9., 10., 10.5],
         [8.5, 9., 10., 11., 12., 13., 14., 14.5],
         [16.5, 17., 18., 19., 20., 21., 22., 22.5],
         [24.5, 25., 26., 27., 28., 29., 30., 30.5],
         [32.5, 33., 34., 35., 36., 37., 38., 38.5],
         [40.5, 41., 42., 43., 44., 45., 46., 46.5],
         [44.5, 45., 46., 47., 48., 49., 50., 50.5]]
    )
    weights = np.asarray(
        [[4., 6., 6., 6., 6., 6., 6., 4.],
         [6., 9., 9., 9., 9., 9., 9., 6.],
         [6., 9., 9., 9., 9., 9., 9., 6.],
         [6., 9., 9., 9., 9., 9., 9., 6.],
         [6., 9., 9., 9., 9., 9., 9., 6.],
         [6., 9., 9., 9., 9., 9., 9., 6.],
         [4., 6., 6., 6., 6., 6., 6., 4.]]
    )
    return smooth, weights


@pytest.fixture
def smooth_step_2_2():
    smooth = np.asarray(
        [[3.45454545, 4.4, 5.34545455, 6.29090909, 7.23636364,
          8.18181818, 9.12727273, 9.12727273],
         [10.38787879, 11.33333333, 12.27878788, 13.22424242, 14.16969697,
          15.11515152, 16.06060606, 16.06060606],
         [17.32121212, 18.26666667, 19.21212121, 20.15757576, 21.1030303,
          22.04848485, 22.99393939, 22.99393939],
         [24.25454545, 25.2, 26.14545455, 27.09090909, 28.03636364,
          28.98181818, 29.92727273, 29.92727273],
         [31.18787879, 32.13333333, 33.07878788, 34.02424242, 34.96969697,
          35.91515152, 36.86060606, 36.86060606],
         [38.12121212, 39.06666667, 40.01212121, 40.95757576, 41.9030303,
          42.84848485, 43.79393939, 43.79393939],
         [45.05454545, 46., 46.94545455, 47.89090909, 48.83636364,
          49.78181818, 50.72727273, 50.72727273]]
    )
    weights = np.asarray(
        [[5.75, 6.125, 6.5, 6.875, 7.25, 7.625, 8., 8.],
         [5.75, 6.125, 6.5, 6.875, 7.25, 7.625, 8., 8.],
         [5.75, 6.125, 6.5, 6.875, 7.25, 7.625, 8., 8.],
         [5.75, 6.125, 6.5, 6.875, 7.25, 7.625, 8., 8.],
         [5.75, 6.125, 6.5, 6.875, 7.25, 7.625, 8., 8.],
         [5.75, 6.125, 6.5, 6.875, 7.25, 7.625, 8., 8.],
         [5.75, 6.125, 6.5, 6.875, 7.25, 7.625, 8., 8.]]
    )
    return smooth, weights


def test_init():
    array = FlaggedArray()
    assert array.data is None
    assert array.dtype == float
    assert array.shape == ()
    assert np.isnan(array.blanking_value)

    array = FlaggedArray(data=np.zeros((10, 10)))
    assert array.shape == (10, 10)
    assert array.dtype == float

    array = FlaggedArray(shape=(5, 5))
    assert array.data.shape == (5, 5)

    array = FlaggedArray(shape=(5, 5), dtype=float, blanking_value=0)
    assert array.blanking_value == 0

    array = FlaggedArray(blanking_value=complex(1, 1))
    assert array.dtype == np.dtype(complex)


def test_copy(dummy_array):
    a = dummy_array
    b = dummy_array.copy()
    assert a == b and a is not b


def test_eq(dummy_array):
    a = dummy_array
    assert a == a
    assert a is not None
    assert a != 1
    other = a.copy()
    assert a == other
    other.set_validating_flags('MASK')
    assert a != other

    other = a.copy()
    other.blanking_value = 1
    assert other != a

    other.blanking_value = None
    assert a != other

    other = a.copy()
    other.data[0, 0] += 1
    assert a != other


def test_check_equal_contents(dummy_array):
    a = dummy_array
    other = FlaggedArray(shape=(2, 2))
    assert not a.check_equal_contents(other)

    other = a.copy()
    other.blanking_value = 1
    assert not a.check_equal_contents(other)

    class FakeArray(object):
        def __init__(self):
            self.data = None
            self.valid = None
            self.flag = None
            self.shape = dummy_array.shape

    other = FakeArray()
    assert not a.check_equal_contents(other)
    other.valid = a.valid
    assert not a.check_equal_contents(other)

    other = a.copy()
    other.data[1, 1] += 1
    assert not a.check_equal_contents(other)

    other = a.copy()
    other.flag[1, 1] += 1
    assert not a.check_equal_contents(other)

    other = FakeArray()
    other.data = a.data
    other.valid = a.valid
    assert not a.check_equal_contents(other)

    assert a.check_equal_contents(a.copy())


def test_nan_blanking(dummy_array):
    a = dummy_array.copy()
    assert a.nan_blanking
    a.blanking_value = 0
    assert not a.nan_blanking
    a._blanking_value = 'a'
    assert not a.nan_blanking


def test_data(dummy_array):
    a = dummy_array.copy()
    assert np.allclose(a.data, a._data)
    a.data = np.zeros((2, 3, 4))
    assert a.shape == (2, 3, 4)


def test_shape(dummy_array):
    a = dummy_array.copy()
    assert a.shape == (7, 8)
    a.shape = (4, 5)
    assert a.shape == (4, 5)
    assert a.data.shape == (4, 5)
    assert np.allclose(a.data, 0)


def test_blanking_value(dummy_array):
    a = dummy_array.copy()
    assert np.isnan(a.blanking_value)
    a.blanking_value = 1
    assert a.blanking_value == 1


def test_size(dummy_array):
    a = dummy_array.copy()
    assert a.size == 56
    a = FlaggedArray()
    assert a.size == 0


def test_ndim(dummy_array):
    assert dummy_array.ndim == 2
    assert FlaggedArray().ndim == 0


def test_valid(dummy_array):
    a = dummy_array.copy()
    a.data[0, 0] = np.nan
    assert not a.valid[0, 0]
    assert a.valid.ravel()[1:].all()


def test_is_valid(dummy_array):
    a = dummy_array.copy()
    assert not FlaggedArray().is_valid()
    a.blanking_value = 0
    v = a.is_valid().ravel()
    assert not v[0] and np.all(v[1:])

    a.set_validating_flags(0)
    a.flag[0, 1] = 1
    v = a.is_valid().ravel()
    assert not np.any(v[:2]) and np.all(v[2:])

    a.set_validating_flags('MASK')
    a.flag[0, 1] = 2
    v = a.is_valid().ravel()
    assert not np.any(v[:2]) and np.all(v[2:])


def test_valid_data(dummy_array):
    assert FlaggedArray().valid_data.size == 0
    a = dummy_array.copy()
    a.data[0, 0] = np.nan
    assert np.allclose(a.valid_data, np.arange(55) + 1)


def test_data_range(dummy_array):
    r = FlaggedArray().data_range
    assert r.min == np.inf and r.max == -np.inf

    a = dummy_array.copy()
    a.data.fill(np.nan)
    r = a.data_range
    assert r.min == np.inf and r.max == -np.inf

    r = dummy_array.data_range
    assert r.min == 0 and r.max == 55


def test_default_blanking_value():
    a = FlaggedArray()
    a.dtype = None
    assert a.dtype is None
    v = a.default_blanking_value()
    assert a.dtype == float
    assert np.isnan(v)
    a.dtype = int
    assert a.default_blanking_value() == -9999
    a.dtype = bool
    assert not a.default_blanking_value()
    a.dtype = complex
    v = a.default_blanking_value()
    assert np.isnan(v) and isinstance(v, complex)
    a.dtype = str
    with pytest.raises(ValueError) as err:
        _ = a.default_blanking_value()
    assert 'Invalid dtype' in str(err.value)


def test_set_data(dummy_array):
    a = dummy_array.copy()
    b = FlaggedArray(dtype=int)
    b.set_data(a.data)
    assert b.dtype == int
    assert np.allclose(a.data, b.data)
    assert np.allclose(b.flag, 0)
    a.data[0, 0] = np.nan
    b.set_data(a)
    assert np.allclose(a.data.ravel()[1:], b.data.ravel()[1:])
    assert b.flag[0, 0] == 1
    assert np.allclose(b.flag.ravel()[1:], 0)

    with pytest.raises(ValueError) as err:
        b.set_data(1)
    assert 'Must supply data as' in str(err.value)


def test_set_data_shape():
    a = FlaggedArray()
    a.set_data_shape((4, 5))
    assert a.shape == (4, 5)
    assert np.allclose(a.data, np.zeros((4, 5)))
    assert np.allclose(a.flag, np.ones((4, 5)))


def test_set_blanking_level():
    a = FlaggedArray()
    a.set_blanking_level(None)
    assert a.blanking_value is None

    a.set_blanking_level('a')
    assert a.blanking_value == 'a'
    a.set_blanking_level(np.nan)
    assert np.isnan(a.blanking_value)
    a.set_blanking_level(np.nan)
    assert np.isnan(a.blanking_value)

    a.set_blanking_level(0)
    assert a.blanking_value == 0
    a.data = np.arange(5)
    v = a.valid
    assert not v[0] and np.all(v[1:])
    a.blanking_value = 0
    assert np.allclose(a.valid, v)
    a.set_blanking_level(1)
    assert np.allclose(a.data, [1, 1, 2, 3, 4])
    assert np.allclose(a.valid, [False, False, True, True, True])


def test_get_size_string(dummy_array):
    assert dummy_array.get_size_string() == '7x8'
    assert FlaggedArray().get_size_string() == '0'


def test_discard_flag(dummy_array):
    a = dummy_array.copy()
    a.flag[0, :5] = 1
    a.flag[1, :5] = 2
    a.flag[2, :5] = 4
    expected = a.data.copy()
    expected[1, :5] = 0
    expected_flag = a.flag.copy()
    expected_flag[1, :5] = 1  # The DISCARD flag
    a.discard_flag(2)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flag)


def test_discard(dummy_array):
    a = dummy_array.copy()
    b = FlaggedArray()
    b.discard()
    assert b.data is None
    a.discard()
    assert np.allclose(a.flag, 1)
    assert np.isnan(a.data).all()

    a = dummy_array.copy()
    indices = np.arange(3), np.arange(3)
    expected = a.data.copy()
    expected_flag = a.flag.copy()
    expected[indices] = 0
    expected_flag[indices] = 1
    a.discard(indices)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flag)


def test_clear(dummy_array):
    a = dummy_array.copy()
    b = FlaggedArray()
    b.clear()
    assert b.data is None
    a.set_flags(1)
    a.clear()
    assert np.allclose(a.data, 0)
    assert np.allclose(a.flag, 0)

    a = dummy_array.copy()
    a.set_flags(1)
    indices = np.arange(3), np.arange(3)
    expected, expected_flags = a.data.copy(), a.flag.copy()
    expected[indices] = 0
    expected_flags[indices] = 0
    a.clear(indices)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flags)


def test_fill(dummy_array):
    a = FlaggedArray()
    a.fill(1)
    assert a.data is None
    a = dummy_array.copy()
    a.set_flags(1)
    a.fill(1)
    assert np.allclose(a.data, 1)
    assert np.allclose(a.flag, 0)
    a.set_flags(1)
    expected = a.data.copy()
    expected_flag = a.flag.copy()
    indices = np.arange(3), np.arange(3)
    expected[indices] = 0
    expected_flag[indices] = 0
    a.fill(0, indices=indices)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flag)


def test_add(dummy_array):
    a = FlaggedArray()
    a.add(1)
    assert a.data is None
    a = dummy_array.copy()
    a.set_flags(1)
    a.add(a)
    assert np.allclose(a.data, dummy_array.data * 2)
    assert np.allclose(a.flag, 0)

    a = dummy_array.copy()
    a.set_flags(1)
    a.add(a, indices=a.valid, factor=2)
    assert np.allclose(a.data, dummy_array.data * 3)
    assert np.allclose(a.flag, 0)

    a = dummy_array.copy()
    a.set_flags(1)
    a.add(a.data)
    assert np.allclose(a.data, dummy_array.data * 2)
    assert np.allclose(a.flag, 0)

    a = dummy_array.copy()
    a.set_flags(1)
    expected = a.data.copy()
    expected_flags = a.flag.copy()
    indices = np.arange(3), np.arange(3)
    expected[indices] += 1
    expected_flags[indices] = 0
    a.add(1)
    assert np.allclose(a.data, dummy_array.data + 1)
    assert np.allclose(a.flag, 0)

    a = dummy_array.copy()
    a.set_flags(1)
    int_indices = np.asarray([indices[0], indices[1]])
    a.add(1, indices=int_indices)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flags)

    a = dummy_array.copy()
    a.set_flags(1)
    bool_indices = np.full(a.shape, False)
    bool_indices[indices] = True
    a.add(1, indices=bool_indices)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flags)


def test_subtract(dummy_array):
    a = dummy_array.copy()
    a.subtract(1)
    assert np.allclose(a.data, dummy_array.data - 1)
    a = dummy_array.copy()
    a.subtract(1, factor=2)
    assert np.allclose(a.data, dummy_array.data - 2)

    a = dummy_array.copy()
    a.set_flags(1)
    mask = (a.data % 2) == 0
    expected = a.data.copy()
    expected_flags = a.flag.copy()
    expected[mask] -= 1
    expected_flags[mask] = 0

    a.subtract(1, indices=mask)
    assert np.allclose(expected, a.data)
    assert np.allclose(expected_flags, a.flag)


def test_scale(dummy_array):
    a = FlaggedArray()
    a.scale(1)
    assert a.data is None
    a = dummy_array.copy()
    a.scale(2)
    assert np.allclose(a.data, dummy_array.data * 2)
    a = dummy_array.copy()
    mask = (a.data % 2) == 0
    a.scale(2, indices=mask)
    expected = dummy_array.data.copy()
    expected[mask] *= 2
    assert np.allclose(a.data, expected)


def test_destroy(dummy_array):
    a = dummy_array.copy()
    a.destroy()
    assert a.shape == (0, 0)


def test_validate(dummy_array):
    a = dummy_array.copy()
    a.data[0, 0] = np.nan
    a.validate()
    assert a.flag[0, 0] == 1
    assert a.data[0, 0] == 0

    class Validator(object):
        def __call__(self, array):
            array.discard((array.data % 2) == 0)

    a = dummy_array.copy()
    expected = a.data.copy()
    expected_flag = a.flag.copy()
    indices = (a.data % 2) == 0
    expected[indices] = 0
    expected_flag[indices] = 1
    a.validate(Validator())
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flag)


def test_find_fixed_index(dummy_array):
    a = FlaggedArray(data=np.arange(10), dtype=float)
    assert np.allclose(a.find_fixed_indices([0, 1, 2]), [0, 1, 2])

    a = dummy_array.copy()
    assert a.find_fixed_indices(-1)[0].size == 0
    assert a.find_fixed_indices(-1, cull=False) == (-1, -1)
    assert a.find_fixed_indices(5) == (0, 5)

    inds = a.find_fixed_indices([1, 11])
    assert np.allclose(inds[0], [0, 1])
    assert np.allclose(inds[1], [1, 3])

    inds = a.find_fixed_indices([1, 11], cull=False)
    assert np.allclose(inds[0], [0, 1])
    assert np.allclose(inds[1], [1, 3])

    inds = a.find_fixed_indices([1, -1])
    assert np.allclose(inds[0], [0])
    assert np.allclose(inds[1], [1])

    inds = a.find_fixed_indices([1, -1], cull=False)
    assert np.allclose(inds[0], [0, -1])
    assert np.allclose(inds[1], [1, -1])


def test_count_flags(dummy_array):
    a = dummy_array.copy()
    a.flag[0, :4] = 1
    assert a.count_flags() == 4
    assert a.count_flags(flag=1) == 4
    assert a.count_flags(flag=2) == 0


def test_get_indices():
    a = FlaggedArray()
    with pytest.raises(NotImplementedError):
        _ = a.get_indices(1)


def test_delete_indices():
    a = FlaggedArray()
    with pytest.raises(NotImplementedError):
        _ = a.delete_indices(1)


def test_insert_blanks():
    a = FlaggedArray()
    with pytest.raises(NotImplementedError):
        _ = a.insert_blanks(1)


def test_merge():
    a = FlaggedArray()
    with pytest.raises(NotImplementedError):
        a.merge(a)


def test_discard_range(dummy_array):
    a = dummy_array.copy()
    r = Range(8, 15)  # Single row of the data should be discarded
    a.discard_range(r)
    assert np.allclose(a.flag, np.atleast_2d([0, 1, 0, 0, 0, 0, 0]).T)
    a = FlaggedArray()
    a.discard_range(r)
    assert a.data is None


def test_restrict_range(dummy_array):
    a = dummy_array.copy()
    r = Range(8, 15)  # Single row of the data should be kept
    a.restrict_range(r)
    assert np.allclose(a.flag, np.atleast_2d([1, 0, 1, 1, 1, 1, 1]).T)
    a = FlaggedArray()
    a.restrict_range(r)
    assert a.data is None


def test_min(dummy_array):
    assert np.isnan(FlaggedArray().min())
    a = dummy_array.copy()
    assert a.min() == 0
    a.data[...] = np.nan
    assert np.isnan(a.min())


def test_argmin(dummy_array):
    a = dummy_array.copy()
    assert a.argmin() == (0, 0)
    a.data[...] = np.nan
    assert a.argmin() == ()
    assert FlaggedArray().argmin() == ()
    a = FlaggedArray(np.arange(5))
    assert a.argmin() == 0


def test_max(dummy_array):
    a = dummy_array.copy()
    assert a.max() == 55
    a.data[...] = np.nan
    assert np.isnan(a.max())
    assert np.isnan(FlaggedArray().max())


def test_argmax(dummy_array):
    a = dummy_array.copy()
    assert a.argmax() == (6, 7)
    a.data[...] = np.nan
    assert a.argmax() == ()
    assert FlaggedArray().argmax() == ()
    a = FlaggedArray(np.arange(5))
    assert a.argmax() == 4


def test_argmaxdev(dummy_array):
    assert FlaggedArray().argmaxdev() == ()
    a = dummy_array.copy()
    assert a.argmaxdev() == (6, 7)
    a.subtract(100)
    assert a.argmaxdev() == (0, 0)
    a.data[...] = np.nan
    assert a.argmaxdev() == ()
    a.data[...] = 0
    a.blanking_value = 0
    assert a.argmaxdev() == ()


def test_mean(dummy_array):
    assert np.isnan(FlaggedArray().mean())
    a = dummy_array.copy()
    m, w = a.mean()
    assert m == 27.5 and w == 56
    weights = np.full(a.shape, 2)
    m, w = a.mean(weights=weights)
    assert m == 27.5 and w == 112


def test_median(dummy_array):
    assert np.isnan(FlaggedArray().median())
    a = dummy_array.copy()
    m, w = a.median()
    assert m == 27.5 and w == 56
    weights = np.full(a.shape, 1.0)
    m, w = a.median(weights=weights)
    assert m == 27 and w == 56  # Uses weighted median Numba function


def test_select(dummy_array):
    assert np.isnan(FlaggedArray().select(0.5))
    a = dummy_array.copy()
    assert a.select(0) == 0
    assert a.select(1) == 55
    with pytest.raises(ValueError) as err:
        _ = a.select(-0.5)
    assert "Fraction must be between 0 and 1" in str(err.value)
    assert a.select(0.5) == 28
    a.data[...] = np.nan
    assert np.isnan(a.select(0.5))


def test_level(dummy_array):
    a = FlaggedArray()
    a.level()
    assert a.data is None
    a = dummy_array.copy()
    weights = np.full(a.shape, 1.0)
    a.level(robust=True, weights=weights)
    assert a.min() == -27 and a.max() == 28
    a = dummy_array.copy()
    a.level(robust=False, weights=weights)
    assert a.min() == -27.5 and a.max() == 27.5
    a = dummy_array.copy()
    a.level(robust=True)
    assert a.min() == -27.5 and a.max() == 27.5
    a = dummy_array.copy()
    a.level(robust=False)
    assert a.min() == -27.5 and a.max() == 27.5


def test_variance(dummy_array):
    assert np.isnan(FlaggedArray().variance())
    a = dummy_array.copy()
    a.data[...] = np.nan
    assert np.isnan(a.variance())
    a = dummy_array.copy()
    assert np.isclose(a.variance(robust=True), 1662.8676, atol=1e-4)
    assert np.isclose(a.variance(robust=False), 1017.5)


def test_rms(dummy_array):
    a = dummy_array.copy()
    assert np.isclose(a.rms(robust=True), 40.7783, atol=1e-4)
    assert np.isclose(a.rms(robust=False), 31.8983, atol=1e-4)


def test_sum(dummy_array):
    assert np.isnan(FlaggedArray().sum())
    a = dummy_array.copy()
    assert a.sum() == 1540
    a.data[...] = np.nan
    assert np.isnan(a.sum())


def test_abs_sum(dummy_array):
    assert np.isnan(FlaggedArray().abs_sum())
    a = dummy_array.copy()
    # Randomly invert data values
    mask = np.random.random(a.shape) > 0.5
    a.scale(-1, indices=mask)
    assert a.abs_sum() == 1540
    a.data[...] = np.nan
    assert np.isnan(a.abs_sum())


def test_mem_correct(dummy_array):
    a = FlaggedArray()
    a.mem_correct(None, np.zeros(2), 0.1)
    assert a.data is None

    a = dummy_array.copy()
    model = a.data - 1
    noise = np.full(a.shape, 0.1)
    lg_factor = 0.1
    a.mem_correct(model, noise, lg_factor)
    assert np.allclose(a.data[0, :3], [0.0230756, 0.9769244, 1.9931058],
                       atol=1e-4)


def test_paste(dummy_array):
    a = dummy_array.copy()
    d0 = a._data
    a.paste(a)
    assert a._data is d0
    a.paste(FlaggedArray())
    assert a._data is d0
    b = FlaggedArray()
    b.paste(a)
    assert np.allclose(a.data, b.data)
    b.data[0, 0] = np.nan
    assert np.allclose(a.flag, 0)
    a.paste(b)
    flag = a.flag.ravel()
    assert flag[0] == 1 and np.allclose(flag[1:], 0)


def test_count_valid_points(dummy_array):
    assert FlaggedArray().count_valid_points() == 0
    assert dummy_array.count_valid_points() == 56


def test_get_index_range(dummy_array):
    assert FlaggedArray().get_index_range().shape == (0, 2)
    assert np.allclose(dummy_array.get_index_range(), [[0, 7], [0, 8]])


def test_fast_smooth(dummy_array, smooth_convolve):
    beam_map = np.ones((3, 3))
    a = dummy_array.copy()
    a.fast_smooth(beam_map, np.full(2, 1))
    assert np.allclose(a.data, smooth_convolve[0])


def test_get_fast_smoothed(dummy_array, smooth_convolve, smooth_step_2_2):
    a = dummy_array.copy()
    beam_map = np.ones((3, 3))
    steps = np.ones(2, dtype=int)
    s = a.get_fast_smoothed(beam_map, steps)
    assert np.allclose(s, smooth_convolve[0])
    s, w = a.get_fast_smoothed(beam_map, steps, get_weights=True)
    assert np.allclose(s, smooth_convolve[0])
    assert np.allclose(w, smooth_convolve[1])

    es, ew = smooth_step_2_2
    s = a.get_fast_smoothed(beam_map, steps + 1)
    assert np.allclose(s, es)
    s, w = a.get_fast_smoothed(beam_map, steps + 1, get_weights=True)
    assert np.allclose(s, es)
    assert np.allclose(w, ew)


def test_smooth(dummy_array, smooth_convolve):
    a = dummy_array.copy()
    beam_map = np.ones((3, 3))
    a.smooth(beam_map)
    assert np.allclose(a.data, smooth_convolve[0])


def test_get_smoothed(dummy_array, smooth_convolve):
    a = dummy_array.copy()
    beam_map = np.ones((3, 3))
    s, w = a.get_smoothed(beam_map)
    assert np.allclose(s, smooth_convolve[0])
    assert np.allclose(w, smooth_convolve[1])


def test_resample_from(dummy_array, smooth_convolve):  # herez
    a = dummy_array.copy()
    b = dummy_array.copy()
    a.clear()
    indices = np.indices(a.shape)
    a.resample_from(b, indices)
    assert np.allclose(a.data, b.data)
    assert np.allclose(a.flag, 0)

    a.clear()
    kernel = np.ones((3, 3))
    a.resample_from(b, indices, kernel=kernel)
    assert np.allclose(a.data, smooth_convolve[0])


def test_direct_resample_from(dummy_array):
    a = dummy_array.copy()
    b = dummy_array.copy()
    indices = np.indices(a.shape)
    a.clear()

    # Shifted 1.5 in x-direction
    offset_indices = indices + np.asarray([0, 1.5])[:, None, None]
    a.direct_resample_from(b, offset_indices)
    assert np.allclose(a.data[:, :6], b.data[:, :6] + 1.5)
    assert np.allclose(a.data[:, 6], b.data[:, 7])
    assert np.allclose(a.data[:, 7], 0)
    assert np.allclose(a.flag[:, :7], 0)
    assert np.allclose(a.flag[:, 7], 1)

    expected = a.data.copy()
    expected_flag = a.flag.copy()
    a.clear()
    a.direct_resample_from(b.data, offset_indices)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag, expected_flag)

    with pytest.raises(ValueError) as err:
        a.direct_resample_from(1, offset_indices)
    assert "Image must be an array" in str(err.value)


def test_value_at(dummy_array):
    a = dummy_array.copy()
    assert a.value_at(np.ones(2, dtype=int)) == 9
    assert a.value_at(np.array([2, 2.5])) == 18.5
    a.data[2, 2] = np.nan
    assert np.isclose(a.value_at(np.array([2, 2.5])), 18.5)


def test_kernel_resmample_from(dummy_array, smooth_convolve):
    a = dummy_array.copy()
    kernel = np.ones((3, 3))
    to_indices = np.indices(a.shape) + np.array([0, 1.5])[:, None, None]
    b = dummy_array.copy()
    weights = b.copy()
    weights.data[...] = 1.0
    a.clear()
    a.kernel_resample_from(b, kernel, to_indices, weights=weights)

    expected = smooth_convolve[0].copy()
    expected[:, 0] += 1
    expected[:, -1] = 0
    expected[:, -2] += 1
    expected[:, 1:-2] += 1.5
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag[:, :-1], 0)
    assert np.allclose(a.flag[:, -1], 1)

    a.clear()
    a.kernel_resample_from(b.data, kernel, to_indices, weights=weights)
    assert np.allclose(a.data, expected)
    assert np.allclose(a.flag[:, :-1], 0)
    assert np.allclose(a.flag[:, -1], 1)


def test_resample_from_skew():
    # This test checks that no transpose operation is performed due to
    # spline coordinates expressed as (x, y), but everything else as (y, x).
    data = np.zeros((51, 51))
    data[25, 25] = 1.0
    kernel = np.zeros((7, 7))

    inds = np.indices(kernel.shape) - 3.0

    dy, dx = abs(inds)
    inds[0] /= 3
    r = np.hypot(*inds)
    r = r.max() - r
    r /= r.max()
    kernel = r  # Vertical bar
    sum_x, sum_y = (dx * kernel).sum(), (dy * kernel).sum()

    skew_kernel = sum_y / sum_x
    assert skew_kernel > 1  # Test is vertical
    a = FlaggedArray(data)
    b = a.copy()
    b.clear()
    to_indices = np.indices(a.shape)
    b.kernel_resample_from(a, kernel, to_indices)
    to_indices = to_indices + 0.1
    b.clear()
    b.kernel_resample_from(a, kernel, to_indices)

    ry, rx = abs(np.indices(b.shape) - 25)
    sum_rx, sum_ry = (rx * b.data).sum(), (ry * b.data).sum()
    skew_resample = sum_ry / sum_rx
    assert skew_resample > 1  # Check still vertical


def test_despike(dummy_array):
    a = dummy_array.copy()
    a.despike(3.0)
    assert np.allclose(a.flag[0], 1)
    assert np.allclose(a.flag[-1], 1)
    assert np.allclose(a.flag[1:-1], 0)


def test_get_neighbor_kernel():
    a = FlaggedArray(shape=(10,))
    assert np.allclose(a.get_neighbor_kernel(), [1, 0, 1])
    a = FlaggedArray(shape=(10, 10))
    assert np.allclose(a.get_neighbor_kernel(),
                       [[0.5, 1., 0.5],
                        [1., 0., 1.],
                        [0.5, 1., 0.5]]
                       )
    a = FlaggedArray(shape=(10, 10, 10))
    assert np.allclose(a.get_neighbor_kernel(),
                       [[[1 / 3, 0.5, 1 / 3],
                         [0.5, 1., 0.5],
                         [1 / 3, 0.5, 1 / 3]],

                        [[0.5, 1., 0.5],
                         [1., 0., 1.],
                         [0.5, 1., 0.5]],

                        [[1 / 3, 0.5, 1 / 3],
                         [0.5, 1., 0.5],
                         [1 / 3, 0.5, 1 / 3]]]
                       )


def test_get_neighbors(dummy_array):
    a = dummy_array.copy()
    a.data[3, 3] = np.nan
    n = a.get_neighbors()
    assert np.allclose(n,
                       [[4, 6, 6, 6, 6, 6, 6, 4],
                        [6, 9, 9, 9, 9, 9, 9, 6],
                        [6, 9, 8, 8, 8, 9, 9, 6],
                        [6, 9, 8, 8, 8, 9, 9, 6],
                        [6, 9, 8, 8, 8, 9, 9, 6],
                        [6, 9, 9, 9, 9, 9, 9, 6],
                        [4, 6, 6, 6, 6, 6, 6, 4]]
                       )


def test_discard_min_neighbors(dummy_array):
    a = dummy_array.copy()
    a.data[3, 3] = np.nan
    a.discard_min_neighbors(9)
    assert np.allclose(a.flag,
                       [[1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1]])


def test_set_validating_flags():
    a = FlaggedArray()
    assert a.validating_flags is None
    a.set_validating_flags('MASK')
    assert a.validating_flags.value == 2


def test_get_info(dummy_array):
    a = dummy_array.copy()
    assert a.get_info() == ['Image Size: 7x8 pixels.']


def test_count_points(dummy_array):
    assert FlaggedArray().count_points() == 0
    assert dummy_array.count_points() == 56


def test_index_of_max(dummy_array):
    a = dummy_array.copy()
    v, i = a.index_of_max()
    assert v == 55 and i == (6, 7)
    data = np.zeros(a.shape)
    data[4, 5] = 1.0
    data[5, 5] = -1.0
    v, i = a.index_of_max(data=data)
    assert v == 1 and i == (4, 5)
    v, i = a.index_of_max(data=data, sign=-1)
    assert v == -1 and i == (5, 5)


def test_get_refined_peak_index(dummy_array):
    a = dummy_array.copy()
    a.data[4, 4] += a.data.max()
    peak_index = (4, 4)
    rp = a.get_refined_peak_index(peak_index)
    assert np.allclose(rp, [4.07272727, 4.00909091], atol=1e-5)
    a.data[4, 5] = np.nan
    rp = a.get_refined_peak_index(peak_index)
    assert np.allclose(rp, [4.07272727, 4], atol=1e-5)
    a.data[0, 0] += a.data[4, 5] + 1
    peak_index = (0, 0)
    rp = a.get_refined_peak_index(peak_index)
    assert np.allclose(rp, [0, 0])
    a.data[6, 7] = a.data[0, 0] + 1
    peak_index = (6, 7)
    rp = a.get_refined_peak_index(peak_index)
    assert np.allclose(rp, [6, 7])


def test_crop(dummy_array):
    a = dummy_array.copy()
    ranges = np.array([[1, 5], [2, 4]])
    a.crop(ranges)
    assert np.allclose(a.data, [[10, 11], [18, 19], [26, 27], [34, 35]])
    assert a.flag.shape == (4, 2)
    assert np.allclose(a.flag, 0)
    a = FlaggedArray()
    a.crop(ranges)
    assert a.data is None


def test_get_cropped(dummy_array):
    a = dummy_array.copy()
    ranges = np.array([[1, 5], [2, 4]])
    b = a.get_cropped(ranges)
    assert b.shape == (4, 2)
    a = FlaggedArray()
    b = a.get_cropped(ranges)
    assert b.data is None
