# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the FORCAST Imaging Reduction class."""

import os
import shutil

from astropy.io import fits
from matplotlib.testing.compare import compare_images
import numpy as np
import pytest

from sofia_redux.toolkit.utilities.fits import set_log_level
try:
    from sofia_redux.pipeline.sofia.exes_quicklook_reduction \
        import EXESQuicklookReduction
    HAS_DRIP = True
except ImportError:
    HAS_DRIP = False
    EXESQuicklookReduction = None


@pytest.fixture
def exes_combspec(tmpdir):
    fname = 'F0001_EX_SPE_12345_EXES_CMB_001.fits'
    header = fits.Header({'INSTRUME': 'EXES',
                          'PRODTYPE': 'combspec',
                          'INSTCFG': 'HI-MED',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(
        data=np.arange(500, dtype=float).reshape((10, 5, 10)), header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.fixture
def exes_mrgordspec(tmpdir):
    fname = 'F0001_EX_SPE_12345_EXES_MRD_001.fits'
    header = fits.Header({'INSTRUME': 'EXES',
                          'PRODTYPE': 'mrgordspec',
                          'INSTCFG': 'HI-MED',
                          'FILENAME': fname})
    hdul = fits.HDUList(fits.PrimaryHDU(
        data=np.arange(50, dtype=float).reshape((5, 10)),
        header=header))
    outname = os.path.join(tmpdir, fname)
    hdul.writeto(outname, overwrite=True)
    return outname


@pytest.mark.skipif('not HAS_DRIP')
class TestEXESQuicklookReduction(object):

    def test_specmap(self, exes_combspec, exes_mrgordspec, tmpdir):
        red = EXESQuicklookReduction()
        red.output_directory = tmpdir
        red.load([exes_combspec, exes_mrgordspec])
        red.load_parameters()

        # expected image file name
        red.step()
        outfile = tmpdir.join('F0001_EX_SPE_12345_EXES_CMB_001.png')
        assert os.path.isfile(outfile)
        outfile = tmpdir.join('F0001_EX_SPE_12345_EXES_MRD_001.png')
        assert os.path.isfile(outfile)

    def test_specmap_ignore_outer(self, tmpdir, capsys,
                                  exes_combspec, exes_mrgordspec):
        red = EXESQuicklookReduction()
        red.output_directory = tmpdir
        red.load([exes_combspec, exes_mrgordspec])
        red.load_parameters()
        parset = red.parameters.current[-1]
        outfile1 = tmpdir.join('F0001_EX_SPE_12345_EXES_CMB_001.png')
        outfile2 = tmpdir.join('F0001_EX_SPE_12345_EXES_MRD_001.png')

        # set ignore outer parameter to turn off
        parset['ignore_outer']['value'] = 0.0
        with set_log_level('DEBUG'):
            red.specmap()
        assert 'Plotting between' not in capsys.readouterr().out
        assert os.path.isfile(outfile1)
        shutil.move(outfile1, tmpdir.join('tmp1.png'))
        assert os.path.isfile(outfile2)
        shutil.move(outfile2, tmpdir.join('tmp2.png'))

        # set ignore outer parameter to ignore outer 20%
        parset['ignore_outer']['value'] = 0.2
        with set_log_level('DEBUG'):
            red.specmap()
        assert 'Plotting between w=2 and w=8' in capsys.readouterr().out
        assert os.path.isfile(outfile1)
        shutil.move(outfile1, tmpdir.join('tmp1a.png'))
        assert os.path.isfile(outfile2)
        shutil.move(outfile2, tmpdir.join('tmp2a.png'))

        # images are the same size, different data plotted
        assert compare_images(tmpdir.join('tmp1.png'),
                              tmpdir.join('tmp1a.png'), 0) is not None
        assert compare_images(tmpdir.join('tmp2.png'),
                              tmpdir.join('tmp2a.png'), 0) is not None

    def test_atran_option(self, tmpdir, exes_combspec):
        red = EXESQuicklookReduction()
        red.output_directory = tmpdir
        red.load([exes_combspec])
        red.load_parameters()
        parset = red.parameters.current[-1]
        outfile = tmpdir.join('F0001_EX_SPE_12345_EXES_CMB_001.png')

        # turn atran off
        parset['atran_plot']['value'] = False
        red.specmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp1.png'))

        # and turn on
        parset['atran_plot']['value'] = True
        red.specmap()
        assert os.path.isfile(outfile)
        shutil.move(outfile, tmpdir.join('tmp2.png'))

        # image is different
        assert compare_images(tmpdir.join('tmp1.png'),
                              tmpdir.join('tmp2.png'), 0) is not None
