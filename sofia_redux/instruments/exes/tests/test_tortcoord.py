# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

import sofia_redux.instruments.exes.tortcoord as et


class TestTortCoord(object):

    def test_create_pixel_array(self):
        nx = 10
        ny = 15

        x, y, xdist, ydist = et._create_pixel_array(nx, ny)

        for out in [x, y, xdist, ydist]:
            assert out.shape == (ny, nx)
        assert np.all(np.diff(x, axis=0)) == 0
        assert np.all(np.diff(x, axis=1)) == 1
        assert np.all(np.diff(y, axis=0)) == 1
        assert np.all(np.diff(y, axis=1)) == 0
        assert np.nanmean(xdist) == 0
        assert np.nanmean(ydist) == 0
