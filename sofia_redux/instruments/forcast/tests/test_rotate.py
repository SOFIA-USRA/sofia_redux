# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.forcast.rotate import (
    rotate, rotate_coordinates_about, rotate_point, rotate_image,
    rotate_image_with_mask)
from sofia_redux.instruments.forcast.merge_centroid import merge_centroid
from sofia_redux.instruments.forcast.tests.resources import npc_testdata


def test_rotate_coordinates_about():
    coordinates = np.indices((4, 5))
    center = np.asarray([2, 1])
    r = rotate_coordinates_about(coordinates, center, 90)

    ey, ex = np.meshgrid(np.arange(5), np.arange(4), indexing='xy')
    ey += 1
    ex = 3 - ex
    assert np.allclose(ex, r[1])
    assert np.allclose(ey, r[0])
    r_inv = rotate_coordinates_about(r, center, 90, inverse=True)
    assert np.allclose(r_inv, coordinates)

    shift = np.ones(2)
    r = rotate_coordinates_about(coordinates, center, 90, shift=shift)
    ey += 1
    ex -= 1
    assert np.allclose(ex, r[1])
    assert np.allclose(ey, r[0])
    r_inv = rotate_coordinates_about(r, center, 90, shift=shift,
                                     inverse=True)
    assert np.allclose(r_inv, coordinates)


def test_rotate_point():
    x = 1
    y = 2
    center = np.zeros(2)
    yr, xr = rotate_point(y, x, center, 90, for_header=False)
    assert np.isclose(yr, 1)
    assert np.isclose(xr, -2)
    yi, xi = rotate_point(yr, xr, center, 90, for_header=False, inverse=True)
    assert np.isclose(xi, x)
    assert np.isclose(yi, y)

    yr, xr = rotate_point(y, x, center, 90, for_header=True)
    assert np.isclose(yr, 1)
    assert np.isclose(xr, 0)
    yi, xi = rotate_point(yr, xr, center, 90, for_header=True, inverse=True)
    assert np.isclose(xi, x)
    assert np.isclose(yi, y)


def test_rotate_image():
    image = np.zeros((4, 5), dtype=float)
    image[2, 3] = 1.0
    rotated = rotate_image(image, 90, cval=np.nan, order=1,
                           center=np.full(2, 2))
    assert np.allclose(rotated[:, np.array([0, 1, 3])], 0)
    assert np.isnan(rotated[:, 4]).all()
    assert np.allclose(rotated[:, 2], [0, 1, 0, 0])

    rotated = rotate_image(image, 90, cval=np.nan, order=3, shift=np.ones(2))
    assert np.allclose(rotated[:3, :3],
                       [[0., 0.36057737, 0.36057737],
                        [0.01623197, 0., 0.],
                        [0., 0.02049922, 0.02049922]], atol=1e-6)
    assert np.isnan(rotated[:, 3:]).all()
    assert np.isnan(rotated[3]).all()


def test_rotate_image_with_mask():
    image = np.zeros((7, 7), dtype=float)
    image[2, 3] = 1.0
    rotated = rotate_image_with_mask(image, 90, order=1)
    mask = np.full(image.shape, False)
    mask[3, 2] = True
    assert np.allclose(rotated[mask], 1)
    assert np.allclose(rotated[~mask], 0)
    image[2, 4] = np.nan
    rotated = rotate_image_with_mask(image, 90, order=1)
    assert np.allclose(rotated[mask], 1)
    assert np.isnan(rotated[2, 2])
    mask[2, 2] = True
    assert np.allclose(rotated[~mask], 0)
    image.fill(1)
    rotated = rotate_image_with_mask(image, 45, order=1)
    mask.fill(True)
    mask[0, 0] = mask[-1, -1] = mask[0, -1] = mask[-1, 0] = False
    assert np.allclose(rotated[mask], 1)
    assert np.isnan(rotated[~mask]).all()
    assert rotate_image_with_mask(None, 45) is None


class TestShift(object):

    def test_errors(self):
        data = npc_testdata()['data']
        assert rotate(data[0], 45) is None
        rot, var = rotate(data, 0, variance=data[0].copy())
        assert np.allclose(rot, data)
        assert var is None

    def test_success(self):
        data = npc_testdata()['data']
        variance = data * 2
        header = npc_testdata()['header']
        angle = header['SKY_ANGL']
        rot, var = rotate(data, angle, header=header, variance=variance)

        # rotation order is different for variance, so pixel
        # values will be a little different, but in the same ballpark
        v = ~(np.isnan(rot) | np.isnan(var))
        assert np.allclose(2 * rot[v], var[v], atol=1, rtol=0.5)

        sx, sy = int(round(header['SRCPOSX'])), int(round(header['SRCPOSY']))
        # Check some rotation has occured
        assert rot[sy, sx] > data[sy, sx]
        assert rot[sy, sx] > np.nanmedian(abs(rot)) * 1000

        # If the test data is good we should have a negative source
        # at a known location after "unrotated" by sky_angl
        dx = int(np.round(header['NODAMP']))
        negsource = rot[sy, sx + dx]
        assert negsource < -np.nanmedian(abs(rot)) * 1000
        # Check map was expanded
        assert rot.size > data.size

    def test_missing(self):
        data = npc_testdata()['data']
        variance = data.copy() * 2
        header = npc_testdata()['header']
        angle = header['SKY_ANGL']
        rot, var = rotate(data, angle, header=header, variance=variance,
                          missing=np.nan)
        assert np.isnan(rot).any()
        assert np.isnan(var).any()
        rot, var = rotate(data, angle, header=header, variance=variance,
                          missing=0)
        assert not np.isnan(rot).any()
        assert not np.isnan(var).any()

    def test_strip_border(self):
        data = npc_testdata()['data']
        variance = data.copy() * 2
        header = npc_testdata()['header']
        image, var = merge_centroid(data, header, variance=variance)

        h0, h1 = header.copy(), header.copy()
        rot, rvar = rotate(image, header['SKY_ANGL'], variance=var,
                           header=h0, strip_border=False)
        rots, rvars = rotate(image, header['SKY_ANGL'], variance=var,
                             header=h1, strip_border=True)

        # stripped size will be larger because it is
        # expanded for differing rotation angles
        assert rot.size < rots.size

        # check that source was found in both cases
        src0 = np.round(np.array([h0['SRCPOSX'], h0['SRCPOSY']])).astype(int)
        src1 = np.round(np.array([h1['SRCPOSX'], h1['SRCPOSY']])).astype(int)

        assert rot[src0[1], src0[0]] > np.nanmedian(abs(rot)) * 1000
        assert rots[src1[1], src1[0]] > np.nanmedian(abs(rots)) * 1000

        assert not np.allclose(src0, src1)
        assert np.allclose(rot[src0[1], src0[0]], rots[src1[1], src1[0]],
                           rtol=0.5)

    def test_center(self):
        # set the center to the CRPIX; this is close
        # to the center of the array, so 'success' test
        # should pass without modification
        data = npc_testdata()['data']
        variance = data * 2
        header = npc_testdata()['header']
        angle = header['SKY_ANGL']
        center = [header['CRPIX1'] - 1, header['CRPIX2'] - 1]
        rot, var = rotate(data, angle, center=center,
                          header=header, variance=variance)

        v = ~(np.isnan(rot) | np.isnan(var))
        assert np.allclose(2 * rot[v], var[v], atol=1, rtol=0.5)

        sx, sy = int(round(header['SRCPOSX'])), int(round(header['SRCPOSY']))
        # Check some rotation has occured
        assert rot[sy, sx] > data[sy, sx]
        assert rot[sy, sx] > np.nanmedian(abs(rot)) * 1000
        # If the test data is good we should have a negative source
        # at a known location after "unrotated" by sky_angl
        dx = int(np.round(header['NODAMP']))
        negsource = rot[sy, sx + dx]
        assert negsource < -np.nanmedian(abs(rot)) * 1000
        # Check map was expanded
        assert rot.size > data.size
