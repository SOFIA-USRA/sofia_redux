# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.toolkit.resampling.resample_utils import (
    check_edge_with_range)

import numpy as np


def test_check_edge_with_range():
    # Define coordinates centered around zero between -1 and 1
    x, y = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    coordinates = np.stack((x.ravel(), y.ravel()))
    reference = np.ones(2)
    mask = np.full(coordinates.shape[1], True)

    # check edge at infinity when threshold = 0 or 1
    threshold = np.zeros(2)
    assert check_edge_with_range(coordinates, reference, mask, threshold)

    threshold = np.ones(2)
    assert check_edge_with_range(coordinates, reference, mask, threshold)

    # both dimensions outside
    reference = np.array([0.7, 0.7])
    threshold = np.array([0.4, 0.4])  # beta = 0.6
    assert not check_edge_with_range(
        coordinates, reference, mask, threshold)

    # Check negative direction
    reference = np.array([-0.7, -0.7])
    threshold = np.array([0.4, 0.4])  # beta = 0.6
    assert not check_edge_with_range(
        coordinates, reference, mask, threshold)

    # both dimensions inside
    reference = np.array([0.5, 0.5])
    assert check_edge_with_range(coordinates, reference, mask, threshold)

    # one out, one in
    reference = np.array([0.5, 0.7])
    assert not check_edge_with_range(
        coordinates, reference, mask, threshold)

    # set edge for outside dimension to infinity to see if passes
    threshold = np.array([0.5, 0.0])
    assert check_edge_with_range(coordinates, reference, mask, threshold)

    # check edge still exists for other dimension
    reference = np.array([0.7, 0.0])
    assert not check_edge_with_range(
        coordinates, reference, mask, threshold)

    # effectively shift center of mass of coordinates by supplying mask
    reference = np.array([0.0, 0.0])
    mask = ((x <= 0) & (y <= 0)).ravel()  # com = [-0.5, -0.5]
    assert not check_edge_with_range(
        coordinates, reference, mask, threshold)

    # Now use a reference that should be inside the range of new com
    reference = np.array([-0.5, -0.5])
    assert check_edge_with_range(coordinates, reference, mask, threshold)
