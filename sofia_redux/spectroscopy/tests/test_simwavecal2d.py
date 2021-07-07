# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.simwavecal2d import simwavecal2d
import pytest


@pytest.fixture
def values():
    shape = 200, 200
    edgecoeffs = np.array(
        [[[90., 0.], [190., 0.]],
         [[50., 0.], [100., 0.]]])
    xranges = np.array([[0., 197.], [0, 194]])
    slith_arc = 100.0
    ds = 0.5
    return shape, edgecoeffs, xranges, slith_arc, ds


def test_failure(values):
    shape, edgecoeffs, xranges, slith_arc, ds = values
    assert simwavecal2d(
        shape[0], edgecoeffs, xranges, slith_arc, ds) is None
    assert simwavecal2d(
        shape, edgecoeffs[:, 0], xranges, slith_arc, ds) is None
    assert simwavecal2d(
        shape, edgecoeffs, xranges[0], slith_arc, ds) is None


def test_success(values):
    shape, edgecoeffs, xranges, slith_arc, ds = values
    result = simwavecal2d(shape, edgecoeffs, xranges, slith_arc, ds)
    wavecal, spatcal, indices = result
    mask = np.isfinite(wavecal)
    expected = np.full_like(mask, False)
    expected[50:90, :195] = True
    expected[90:190, :198] = True
    assert np.allclose(mask, expected)
    assert np.nanmax(wavecal) == 197
    assert np.nanmax(indices[0]['y']) == 189
    assert np.nanmin(indices[0]['y']) == 90
    assert np.nanmax(indices[0]['x']) == 197
    assert np.nanmin(indices[0]['x']) == 0
    assert np.nanmax(indices[1]['y']) == 99
    assert np.nanmin(indices[1]['y']) == 50
    assert np.nanmax(indices[1]['x']) == 194
    assert np.nanmin(indices[1]['x']) == 0
    assert np.allclose(spatcal[99, :195], 98)
    assert np.allclose(spatcal[99, 195:198], 9)
