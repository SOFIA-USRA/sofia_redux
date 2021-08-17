# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.flitecam.clipimg import clipimg
from sofia_redux.instruments.flitecam.tests.resources import raw_testdata


class TestClipimg(object):

    def test_success(self):
        hdul = raw_testdata()
        hdr = hdul[0].header.copy()
        orig = hdul[0].data.copy()
        shp = orig.shape

        # datasec matches data: no change
        ds = [0, shp[1], 0, shp[0]]
        clip = clipimg(hdul, ds)
        assert clip[0].data.shape == shp
        assert np.allclose(clip[0].data, orig)
        assert clip[0].header['CRPIX1'] == hdr['CRPIX1']
        assert clip[0].header['CRPIX2'] == hdr['CRPIX2']

        # datasec smaller than data
        ds = [10, shp[1] - 10, 10, shp[0] - 10]
        clip = clipimg(hdul, ds)
        assert clip[0].data.shape == (shp[0] - 20, shp[1] - 20)
        assert np.allclose(clip[0].data, orig[ds[2]:ds[3],
                                              ds[0]:ds[1]])
        assert clip[0].header['CRPIX1'] == hdr['CRPIX1'] - ds[0]
        assert clip[0].header['CRPIX2'] == hdr['CRPIX2'] - ds[2]

    def test_error(self, capsys):
        # bad datasec
        hdul = raw_testdata()
        with pytest.raises(ValueError):
            clipimg(hdul, None)

        hdul = raw_testdata()
        with pytest.raises(ValueError):
            clipimg(hdul, [1, 2, 3])

        hdul = raw_testdata()
        with pytest.raises(ValueError):
            clipimg(hdul, [1, 2, 'a', 4])

        hdul = raw_testdata()
        with pytest.raises(ValueError):
            clipimg(hdul, [-40, 10, 400, 300])

        # okay
        hdul = raw_testdata()
        clipimg(hdul, [1, 20, 3, 40])

        # bad CRPIX
        hdul = raw_testdata()
        hdul[0].header['CRPIX1'] = 'bad'
        with pytest.raises(ValueError):
            clipimg(hdul, [1, 20, 3, 40])

        # missing CRPIX
        hdul = raw_testdata()
        del hdul[0].header['CRPIX1']
        with pytest.raises(ValueError):
            clipimg(hdul, [1, 20, 3, 40])
