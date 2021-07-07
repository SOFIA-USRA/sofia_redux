# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FORCAST Imaging Reduction class."""

import os

from astropy.io import fits
import numpy as np
import pytest

try:
    from sofia_redux.pipeline.sofia.flitecam_imgmap_reduction \
        import FLITECAMImgmapReduction
    from sofia_redux.pipeline.sofia.flitecam_specmap_reduction \
        import FLITECAMSpecmapReduction
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    FLITECAMImgmapReduction = None
    FLITECAMSpecmapReduction = None


@pytest.fixture
def flitecam_image(tmpdir):
    fname = 'F0001_FC_IMA_12345_FLTPa_CAL_001.fits'
    header = fits.Header({'INSTRUME': 'FLITECAM',
                          'PRODTYPE': 'calibrated',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.zeros((2, 10, 10)),
                                        header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.fixture
def flitecam_grism(tmpdir):
    fname = 'F0001_FC_GRI_12345_FLTC2LMFLTSS20_CMB_001.fits'
    header = fits.Header({'INSTRUME': 'FLITECAM',
                          'PRODTYPE': 'combspec',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.zeros((5, 10)),
                                        header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.mark.skipif('not HAS_DRIP')
class TestFLITECAMImgmapReduction(object):

    def test_imgmap(self, flitecam_image, tmpdir):
        red = FLITECAMImgmapReduction()
        red.output_directory = tmpdir
        red.load(flitecam_image)
        red.load_parameters()

        # expected image file name
        outfile = tmpdir.join('F0001_FC_IMA_12345_FLTPa_CAL_001.png')
        red.step()
        assert os.path.isfile(outfile)


@pytest.mark.skipif('not HAS_DRIP')
class TestFLITECAMSpecmapReduction(object):

    def test_specmap(self, flitecam_grism, tmpdir):
        red = FLITECAMSpecmapReduction()
        red.output_directory = tmpdir
        red.load(flitecam_grism)
        red.load_parameters()

        # expected image file name
        outfile = tmpdir.join('F0001_FC_GRI_12345_FLTC2LMFLTSS20_CMB_001.png')
        red.step()
        assert os.path.isfile(outfile)
