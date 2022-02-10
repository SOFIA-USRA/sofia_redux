# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import warnings

import sofia_redux.scan.flags.flag_numba_functions as nf


def test_set_flags():
    flag_array = np.arange(6)
    nf.set_flags(flag_array, 2)
    assert np.allclose(flag_array, [2, 3, 2, 3, 6, 7])

    flag_array = np.arange(6)
    nf.set_flags(flag_array, 2, indices=np.array([0, 1]))
    assert np.allclose(flag_array, [2, 3, 2, 3, 4, 5])

    flag_array = np.arange(6).reshape((2, 3))
    indices = np.nonzero((flag_array == 0) | (flag_array == 5))
    nf.set_flags(flag_array, 2, indices=indices)
    assert np.allclose(flag_array, [[2, 1, 2], [3, 4, 7]])


def test_unflag():
    flag_array = np.arange(6)
    nf.unflag(flag_array)
    assert np.allclose(flag_array, 0)

    flag_array = np.arange(6)
    nf.unflag(flag_array, 2)
    assert np.allclose(flag_array, [0, 1, 0, 1, 4, 5])

    flag_array = np.arange(6).reshape((2, 3))
    indices = np.nonzero(np.isfinite(flag_array))
    nf.unflag(flag_array, indices=indices)
    assert np.allclose(flag_array, 0)

    flag_array = np.arange(6).reshape((2, 3))
    nf.unflag(flag_array, 2, indices=indices)
    assert np.allclose(flag_array, [[0, 1, 0], [1, 4, 5]])


def test_flatten_nd_indices():
    indices = tuple(x.ravel() for x in np.indices((2, 3)))
    flat_indices = nf.flatten_nd_indices(indices, (2, 3))
    assert np.allclose(flat_indices, np.arange(6))


def test_is_flagged():
    flag_array = np.arange(6)
    assert np.allclose(nf.is_flagged(flag_array), [False] + [True] * 5)
    assert np.allclose(nf.is_flagged(flag_array, 0), [True] + [False] * 5)
    assert np.allclose(nf.is_flagged(flag_array, 1, exact=True),
                       [0, 1, 0, 0, 0, 0])
    assert np.allclose(nf.is_flagged(flag_array, 1), [0, 1, 0, 1, 0, 1])


def test_is_unflagged():
    flag_array = np.arange(6)
    assert np.allclose(nf.is_unflagged(flag_array), [True] + [False] * 5)
    assert np.allclose(nf.is_unflagged(flag_array, 0), [False] + [True] * 5)
    assert np.allclose(nf.is_unflagged(flag_array, 1, exact=True),
                       [1, 0, 1, 1, 1, 1])
    assert np.allclose(nf.is_unflagged(flag_array, 1), [1, 0, 1, 0, 1, 0])


def test_get_mem_correction():
    np.random.seed(0)
    data = np.random.random(100) - 0.5
    noise = np.random.random(100) / 5

    def expected(d, n, lm, v, m):
        if m is None:
            m = np.zeros(d.shape)
        if v is None:
            v = np.full(d.shape, True)
        is_valid = v & np.isfinite(m) & np.isfinite(d) & np.isfinite(n)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mem = np.sign(d) * n * lm * np.log(np.sqrt(d ** 2 + n ** 2)
                                               / np.sqrt(m ** 2 + n ** 2))
        mem[~is_valid] = 0.0
        return mem

    result = nf.get_mem_correction(data, noise, multiplier=0.1)
    assert np.allclose(result, expected(data, noise, 0.1, None, None))

    valid = np.full(data.shape, True)
    valid[10] = False
    data[11] = np.nan
    noise[12] = np.nan
    model = np.full(data.shape, 0.05)
    model[13] = np.nan
    result = nf.get_mem_correction(data, noise, multiplier=0.1, valid=valid,
                                   model=model)
    assert np.allclose(result, expected(data, noise, 0.1, valid, model))


def test_set_new_blank_value():
    test = np.random.random(100)
    test[10:15] = np.inf
    nf.set_new_blank_value(test, np.inf, np.nan)
    assert np.isnan(test[10:15]).all()
    nf.set_new_blank_value(test, test[0], np.nan)
    assert np.isnan(test[0])
    old = test.copy()
    nf.set_new_blank_value(test, test[0], None)
    assert np.allclose(old, test, equal_nan=True)
