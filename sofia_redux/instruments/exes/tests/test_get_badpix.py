# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.exes import get_badpix as gbp
from sofia_redux.instruments.exes.readhdr import readhdr


class TestGetBadpix(object):

    def test_get_badpix(self, capsys):
        # default everything: returns None
        header = fits.Header()
        header['FDATE'] = 15.0401
        mask = gbp.get_badpix(header)
        assert mask is None

        # add basic defaults: now returns earliest bpm
        header = readhdr(header, check_header=False)
        mask = gbp.get_badpix(header)
        assert mask.shape == (1024, 1032)
        capt = capsys.readouterr()
        assert 'Using bad pixel mask' in capt.out
        assert 'right' in capt.out
        default = mask.copy()

        # clip 8 reference pixels on right
        mask = gbp.get_badpix(header, clip_reference=True)
        assert mask.shape == (1024, 1024)
        assert np.allclose(mask, default[:, :1024])

        # apply detsec
        header['DETSEC'] = '[11,40,21,60]'
        mask = gbp.get_badpix(header, apply_detsec=True)
        assert mask.shape == (40, 30)
        assert np.allclose(mask, default[20:60, 10:40])

        # apply detsec
        header['DETSEC'] = '[11,40,21,60]'
        mask = gbp.get_badpix(header, apply_detsec=True)
        assert mask.shape == (40, 30)
        assert np.allclose(mask, default[20:60, 10:40])

        # clip and detsec
        header['DETSEC'] = '[1011,1024,1,1024]'
        mask = gbp.get_badpix(header, clip_reference=True, apply_detsec=True)
        assert mask.shape == (1024, 14)
        assert np.allclose(mask, default[:, 1010:1024])

    def test_good_file(self, capsys):
        header = fits.Header()
        header['BPM'] = 'bpm_2015.02.06_right.fits'
        assert gbp.get_badpix(header).shape == (1024, 1032)

    def test_bad_file(self, capsys):
        header = fits.Header()
        header['BPM'] = 'bad'
        assert gbp.get_badpix(header) is None
