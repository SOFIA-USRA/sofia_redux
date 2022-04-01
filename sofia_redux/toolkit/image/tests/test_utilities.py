# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np

from sofia_redux.toolkit.image.utilities import (
    to_ndimage_mode, clip_output, map_coordinates)


def test_to_ndimage_mode():
    assert to_ndimage_mode('edge') == 'nearest'
    assert to_ndimage_mode('symmetric') == 'reflect'
    assert to_ndimage_mode('reflect') == 'mirror'
    with pytest.raises(ValueError) as err:
        _ = to_ndimage_mode('foo')
    assert 'Unknown mode' in str(err.value)


def test_clip_output():
    original = np.arange(25, dtype=float).reshape(5, 5)
    warped = original.copy()
    warped[0, 0] = -1
    warped[0, 1] = 100
    mode = 'constant'
    cval = 2
    w0 = warped.copy()
    clip_output(original, warped, mode, cval, False)
    assert np.allclose(warped, w0)

    clip_output(original, warped, mode, cval, True)
    inds = np.nonzero(original != warped)
    assert warped[inds] == 24
    assert inds[0][0] == 0
    assert inds[1][0] == 1

    warped = original.copy()
    warped[1, 1] = np.nan
    cval = np.nan
    w0 = warped.copy()
    clip_output(original, warped, mode, cval, True)
    assert np.allclose(w0, warped, equal_nan=True)

    warped[1, 2] = 100
    clip_output(original, warped, 'edge', cval, True)
    assert warped[1, 2] == 24
    assert np.isnan(warped[1, 1])


def test_map_coordinates():
    image = np.zeros((5, 4), dtype=float)
    image[1, 2] = 1.0
    coordinates = np.indices(image.shape)
    ct = coordinates[::-1]  # transpose
    mapped = map_coordinates(image, ct, mode='edge', cval=0.0)

    mask = np.full(image.shape, False)
    mask[2, 1] = True
    assert np.allclose(mapped[mask], 1)
    assert np.allclose(mapped[~mask], 0)

    outputs = [None, image.copy()]
    for output in outputs:
        m = mask.copy()
        mapped = map_coordinates(image, ct, mode='constant', cval=np.nan,
                                 output=output)
        assert np.allclose(mapped[m], 1)
        assert np.isnan(mapped[4]).all()
        m[4] = True
        assert np.allclose(mapped[~m], 0)

    output = mapped.astype(int)
    new = map_coordinates(image, ct, mode='constant', cval=-1,
                          output=output, clip=False)
    assert new.dtype == int
    assert np.allclose(new[:2], 0)
    assert np.allclose(new[2], [0, 1, 0, 0])
    assert np.allclose(new[3], 0)
    assert np.allclose(new[4], -1)
