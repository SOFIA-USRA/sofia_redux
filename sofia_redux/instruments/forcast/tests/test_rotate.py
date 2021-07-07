# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.forcast.rotate import rotate
from sofia_redux.instruments.forcast.merge_centroid import merge_centroid
from sofia_redux.instruments.forcast.tests.resources import npc_testdata


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
