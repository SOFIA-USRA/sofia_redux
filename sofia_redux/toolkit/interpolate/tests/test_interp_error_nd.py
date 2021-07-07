# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.toolkit.interpolate.interpolate import interp_error_nd


def test_expected():
    y, x = np.mgrid[:10, :10]
    xy = np.stack([x.ravel() * 1.0, y.ravel()]).T
    error = np.ones(x.size)
    xyout = np.asarray([[1.0, 2.0], [1.5, 2.5]])

    # test all cval if outside range
    assert np.isnan(interp_error_nd(xy, error, xyout - 100)).all()

    # expected values with vector input
    result = interp_error_nd(xy, error, xyout)
    assert np.allclose(result, [1, np.sqrt(3.5)])

    # expected output with scalar input
    result = interp_error_nd(xy, 1.0, xyout)
    assert np.allclose(result, [1, np.sqrt(3.5)])

    # expected output with no interpolation
    result = interp_error_nd(xy, 1.0, xyout.astype(int))
    assert np.allclose(result, 1)


def test_3d():
    z, y, x = np.mgrid[:10, :10, :10]
    xyz = np.stack([x.ravel() * 1.0, y.ravel(), z.ravel()]).T
    error = np.ones(x.size)
    out = np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    result = interp_error_nd(xyz, error, out)
    assert np.allclose(result, [1, np.sqrt(4.5)])
