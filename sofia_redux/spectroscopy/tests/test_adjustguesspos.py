# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.adjustguesspos import adjustguesspos
import pytest


@pytest.fixture
def data():
    shape = 114, 114
    edgecoeffs = np.array(
        [[[5., 0.1], [35., 0.1]],
         [[35., 0.1], [65., 0.1]],
         [[65., 0.1], [95., 0.1]]])
    xranges = np.array([[10, 40], [40, 70], [70, 100]])
    ordermask = np.zeros(shape, dtype=int)
    ordermask[5:35, 10:40] = 1
    ordermask[35:65, 40:70] = 2
    ordermask[65:95, 70:100] = 3
    flat = np.zeros(shape)
    flat[15:45, 10:40] = 1
    flat[45:75, 40:70] = 1
    flat[75:105, 70:100] = 1

    return edgecoeffs, xranges, flat, ordermask


def test_failure(data):
    edgecoeffs, xranges, flat, ordermask = data
    assert adjustguesspos(edgecoeffs[0], xranges, flat, ordermask) is None
    assert adjustguesspos(edgecoeffs, xranges[0], flat, ordermask) is None
    assert adjustguesspos(edgecoeffs, xranges, flat[0], ordermask) is None
    assert adjustguesspos(edgecoeffs, xranges, flat, ordermask[0]) is None
    assert adjustguesspos(edgecoeffs, xranges[:-1], flat, ordermask[0]) is None


def test_success(data):
    edgecoeffs, xranges, flat, ordermask = data
    guess, xout = adjustguesspos(edgecoeffs, xranges, flat, ordermask)
    assert np.allclose(guess, [[32, 25], [66, 55], [98, 85]])
    assert np.allclose(xout, [[10, 40], [40, 70], [70, 99]])


def test_default(data):
    edgecoeffs, xranges, flat, ordermask = data
    guess, xout = adjustguesspos(edgecoeffs, xranges, flat,
                                 ordermask, default=True)
    assert np.allclose(guess, [[22, 25], [56, 55], [88, 85]])
    assert np.allclose(xout, xranges)


def test_off_buffer(data):
    edgecoeffs, xranges, flat, ordermask = data
    guess, xout = adjustguesspos(edgecoeffs, xranges, flat,
                                 ordermask, ybuffer=10)
    assert np.allclose(guess, [[32, 25], [66, 55], [98, 85]])
    assert np.allclose(xout, [[10, 40], [40, 70], [-1, -1]])


def test_orders(data):
    edgecoeffs, xranges, flat, ordermask = data

    # specify orders 2 and 3; edgecoeffs and xranges must match
    guess, xout = adjustguesspos(edgecoeffs[1:], xranges[1:], flat, ordermask,
                                 orders=['2', '3'])
    assert np.allclose(guess, [[66, 55], [98, 85]])
    assert np.allclose(xout, [[40, 70], [70, 99]])
