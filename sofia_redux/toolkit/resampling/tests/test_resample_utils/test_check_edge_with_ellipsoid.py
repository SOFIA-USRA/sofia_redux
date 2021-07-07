# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    check_edge_with_ellipsoid)

import numpy as np


def test_check_edge_with_ellipsoid():
    # Define coordinates centered around zero between -1 and 1
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    coordinates = np.stack((x.ravel(), y.ravel()))

    reference = np.ones(2)
    mask = np.full(coordinates.shape[1], True)

    # check edge at infinity when threshold = 0 or 1
    threshold = np.ones(2)
    assert check_edge_with_ellipsoid(coordinates, reference, mask, threshold)

    threshold = np.zeros(2)
    assert check_edge_with_ellipsoid(coordinates, reference, mask, threshold)

    # both dimensions outside
    threshold = np.full(2, 0.5)
    assert not check_edge_with_ellipsoid(
        coordinates, reference, mask, threshold)

    # both dimensions inside
    threshold = np.full(2, 0.25)
    reference = np.full(2, 0.5)
    assert check_edge_with_ellipsoid(coordinates, reference, mask, threshold)

    # one out, one in
    threshold = np.array([0.5, 0.5])
    reference = np.array([0.0, 0.6])
    assert not check_edge_with_ellipsoid(
        coordinates, reference, mask, threshold)

    # set edge for outside dimension to infinity to see if passes
    threshold = np.array([0.5, 0.0])
    reference = np.array([0.4, 1.0])
    assert check_edge_with_ellipsoid(coordinates, reference, mask, threshold)

    # check edge still exists for other dimension
    threshold = np.array([0.5, 0.0])
    reference = np.array([0.6, 0.0])
    assert not check_edge_with_ellipsoid(
        coordinates, reference, mask, threshold)

    # effectively shift center of mass of coordinates by supplying mask
    threshold = np.full(2, 0.6)
    reference = np.zeros(2)
    mask = ((x <= 0) & (y <= 0)).ravel()  # com = [-0.5, -0.5]
    assert not check_edge_with_ellipsoid(
        coordinates, reference, mask, threshold)

    # Now use a reference that should be inside the range of new com
    reference = np.array([-0.5, -0.5])
    assert check_edge_with_ellipsoid(coordinates, reference, mask, threshold)
