# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import no_fit_solution

import numpy as np
import pytest


@pytest.fixture
def random_arrays():
    n_sets, n_fits = 3, 20
    shape = n_sets, n_fits
    fit_out = np.random.random(shape)
    error_out = np.random.random(shape)
    counts_out = np.random.random(shape)
    weights_out = np.random.random(shape)
    distance_weights_out = np.random.random(shape)
    rchi2_out = np.random.random(shape)
    offset_variance_out = np.random.random(shape)
    return (fit_out, error_out, counts_out, weights_out,
            distance_weights_out, rchi2_out, offset_variance_out)


def test_no_fit_solution(random_arrays):
    (fit, error, counts, weights,
     distance_weights, rchi2, offset_variance) = random_arrays

    set_index = 1
    point_index = 5

    no_fit_solution(set_index, point_index, fit, error, counts, weights,
                    distance_weights, rchi2, offset_variance)
    assert np.isnan(fit[set_index, point_index])
    assert np.isnan(error[set_index, point_index])
    assert counts[set_index, point_index] == 0
    assert weights[set_index, point_index] == 0
    assert distance_weights[set_index, point_index] == 0
    assert np.isnan(rchi2[set_index, point_index])
    assert np.isnan(offset_variance[set_index, point_index])


def test_options(random_arrays):
    (fit, error, counts, weights,
     distance_weights, rchi2, offset_variance) = random_arrays

    set_index = 1
    point_index = 5
    e0 = error[set_index, point_index]
    c0 = counts[set_index, point_index]
    w0 = weights[set_index, point_index]
    d0 = distance_weights[set_index, point_index]
    r0 = rchi2[set_index, point_index]
    o0 = offset_variance[set_index, point_index]

    no_fit_solution(set_index, point_index, fit, error, counts, weights,
                    distance_weights, rchi2, offset_variance,
                    get_error=False, get_counts=False, get_weights=False,
                    get_distance_weights=False, get_rchi2=False,
                    get_offset_variance=False, cval=-1)
    assert fit[set_index, point_index] == -1
    assert error[set_index, point_index] == e0
    assert counts[set_index, point_index] == c0
    assert weights[set_index, point_index] == w0
    assert distance_weights[set_index, point_index] == d0
    assert rchi2[set_index, point_index] == r0
    assert offset_variance[set_index, point_index] == o0
