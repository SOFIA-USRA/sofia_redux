# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.modeling.models import Gaussian1D
import pytest
from sofia_redux.spectroscopy.findorders import findorders


@pytest.fixture
def data():
    image = np.zeros((256, 256))
    yy, xx = np.mgrid[:256, :256]

    def trace_order1(x):  # coeffs = [100, 0, 0.001, 0]
        return 100 + 0.001 * (x - 128) ** 2

    def trace_order2(x):  # coeffs = [150, 0, -0.001, 0]
        return 150 - 0.001 * (x - 128) ** 2

    def trace_order3(x):  # coeffs = [200, 0.25, 0, 0] (goes off image)
        return 200 + x / 4

    def trace_order4(x):  # coeffs = [-25, 0.25, 0, 0] (goes off image)
        return -25 + x / 4

    model = Gaussian1D(amplitude=1.0, mean=0, stddev=2)
    image += model(yy - trace_order1(xx[0]))
    image += model(yy - trace_order2(xx[0]))
    image += model(yy - trace_order3(xx[0]))
    image += model(yy - trace_order4(xx[0]))

    guesspos = np.zeros((4, 2))
    guesspos[0] = 128, 100
    guesspos[1] = 128, 150
    guesspos[2] = 100, 225
    guesspos[3] = 200, 25
    return image, guesspos


def test_failures(data):
    image, guesspos = data
    with pytest.raises(ValueError):
        findorders(image, guesspos[:, :1])
    with pytest.raises(ValueError):
        findorders(image[0], guesspos)
    with pytest.raises(ValueError):
        findorders(image, guesspos, sranges=np.zeros((2, 1)))
    with pytest.raises(ValueError):
        findorders(image, guesspos, sranges=np.zeros((1, 2)))


def test_all_orders(data):
    image, guesspos = data
    edgecoeffs, xranges = findorders(image, guesspos)
    assert edgecoeffs.shape[:2] == (4, 2)
    assert xranges.shape == (4, 2)
    assert np.allclose(xranges[0], [0, 255])
    assert np.allclose(xranges[1], [0, 255])
    assert xranges[2, 0] == 0
    assert xranges[2, 1] < 255
    assert xranges[3, 0] > 0
    assert xranges[3, 1] == 255


def test_single_order(data):
    image, guesspos = data
    edgecoeffs, xranges = findorders(image, guesspos[2], degree=1)
    assert edgecoeffs.shape == (1, 2, 2)
    assert np.allclose(edgecoeffs[0, :, 0], 200, atol=5)
    assert np.allclose(edgecoeffs[0, :, 1], 0.25, atol=0.1)


def test_srange(data, capsys):
    image, guesspos = data

    # 1D srange, otherwise matches single_order test
    edgecoeffs, xranges = findorders(image, guesspos[2], degree=1,
                                     sranges=[20, 230])
    assert edgecoeffs.shape == (1, 2, 2)

    # 2D srange
    edgecoeffs, xranges = findorders(image, guesspos,
                                     sranges=[[20, 230]] * len(guesspos))
    assert edgecoeffs.shape[:2] == (4, 2)

    # 3D srange -- throws error
    with pytest.raises(ValueError) as err:
        findorders(image, guesspos, sranges=[[20, 230, 500]] * len(guesspos))
    assert 'sranges must have 2 elements per order' in str(err)


def test_slit_heights(data):
    image, guesspos = data
    default_edge, xranges = findorders(image, guesspos[0], degree=2)
    assert np.allclose(xranges, [0, 255])

    # test a minor height restriction
    edgecoeffs, xranges = findorders(image, guesspos[0], degree=2,
                                     slith_range=[0, 10])
    assert np.allclose(xranges, [0, 255])
    # higher upper edge for default, since height is not restricted
    assert edgecoeffs[0, 1, 0] < default_edge[0, 1, 0]

    # test a too-restrictive height restriction -- all values nan/0
    edgecoeffs, xranges = findorders(image, guesspos[0], degree=2,
                                     slith_range=[0, 1])
    assert np.isnan(edgecoeffs).all()
    assert np.allclose(xranges, 0)
