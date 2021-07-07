# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.resampling.resample_utils import (
    scale_coordinates, scale_forward_scalar, scale_reverse_scalar,
    scale_forward_vector, scale_reverse_vector)


def test_scale_coordinates():
    random = np.random.RandomState(41)
    dvec = random.rand(3, 32) + 1
    dscalar = dvec[:, 0]

    scale = np.arange(3) + 1.0
    offset = np.arange(3) + 10.0

    # scale_forward_scalar
    x = scale_coordinates(dscalar, scale, offset, reverse=False)
    assert np.allclose(x, (dscalar - offset) / scale)

    # scale_reverse_scalar
    x = scale_coordinates(dscalar, scale, offset, reverse=True)
    assert np.allclose(x, dscalar * scale + offset)

    # scale_forward_vector
    x = scale_coordinates(dvec, scale, offset, reverse=False)
    assert np.allclose(x, (dvec - offset[:, None]) / scale[:, None])

    # scale_reverse_vector
    x = scale_coordinates(dvec, scale, offset, reverse=True)
    assert np.allclose(x, dvec * scale[:, None] + offset[:, None])


def test_forward_scalar():
    coordinates = np.arange(3, dtype=np.float64) + 1
    offset = np.full(3, 10.0)
    scale = np.full(3, 2.0)
    assert np.allclose(scale_forward_scalar(coordinates, scale, offset),
                       [-4.5, -4, -3.5])


def test_forward_vector():
    coordinates = np.repeat(np.arange(3, dtype=np.float64)[None] + 1,
                            3, axis=0)
    offset = np.full(3, 10.0)
    scale = np.full(3, 2.0)
    result = scale_forward_vector(coordinates, scale, offset)
    assert np.allclose(result, [[-4.5, -4, -3.5]])
    assert result.shape == (3, 3)


def test_reverse_scalar():
    coordinates = np.arange(3, dtype=np.float64) + 1
    offset = np.full(3, 10.0)
    scale = np.full(3, 2.0)
    result = scale_reverse_scalar(coordinates, scale, offset)
    assert np.allclose(result, [12, 14, 16])


def test_reverse_vector():
    coordinates = np.repeat(np.arange(3, dtype=np.float64)[None] + 1,
                            3, axis=0)
    offset = np.full(3, 10.0)
    scale = np.full(3, 2.0)
    result = scale_reverse_vector(coordinates, scale, offset)
    assert np.allclose(result, [[12, 14, 16]])
    assert result.shape == (3, 3)
