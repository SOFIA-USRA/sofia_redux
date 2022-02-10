# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepscanmapflat import StepScanMapFlat
from sofia_redux.instruments.hawc.steps.stepscanmapflat import Reduction as SMR
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, scan_raw_data


class TestScanMapFlat(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = scan_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            step = StepScanMapFlat()

            out = step(df, options="rounds=1")
            assert isinstance(out, DataFits)
            assert len(out.imgnames) == 4
            assert 'R ARRAY GAIN' in out.imgnames
            assert 'T ARRAY GAIN' in out.imgnames
            assert 'R BAD PIXEL MASK' in out.imgnames
            assert 'T BAD PIXEL MASK' in out.imgnames

    def test_errors(self, tmpdir, capsys, mocker):
        # make some minimal input data
        df = DataFits()
        df.imageset(np.zeros((10, 10)), imagename='PRIMARY IMAGE')
        df.imageset(np.zeros((10, 10)), imagename='NOISE')
        df.imageset(np.zeros((10, 10)), imagename='EXPOSURE')
        df.setheadval('TELVPA', 270.0)
        df.setheadval('BUNIT', 'counts')
        fname = str(tmpdir.join('testdata.fits'))
        df.filename = fname
        df.save()

        # mock scanmap to return faster
        mocker.patch.object(SMR, 'run', return_value=None)

        with tmpdir.as_cwd():
            step = StepScanMapFlat()
            # default -- no output saved by mock scanmap, so will raise error
            with pytest.raises(ValueError):
                step(df)
            capt = capsys.readouterr()
            assert 'Unable to open scan map output file' in capt.err

            # if df.filename is not on disk, it will check for
            # df.rawname
            df.rawname = fname
            df.filename = 'badfile.fits'
            with pytest.raises(ValueError):
                step(df)
            step.arglist = {}
            capt = capsys.readouterr()
            assert os.path.basename(fname) in capt.out
            assert 'badfile.fits' not in capt.out

    def test_scanmapflat_frame_clip(self, tmpdir, capsys):
        """Test scan map frame clipping options in flat mode."""
        # make some data to test
        nf = 80
        hdul = scan_raw_data(nframe=nf)
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        inp = DataFits(ffile)

        with tmpdir.as_cwd():
            step = StepScanMapFlat()

            # run with all frames used, turning off some other
            # clipping options
            options = 'downsample=1 blacklist=vclip,fillgaps ' \
                      'shift=0 chopper.shift=0 rounds=1 q'
            step(inp, use_frames='', options=options)
            capt = capsys.readouterr()
            assert f'Reading {nf} frames' in capt.out
            assert f'{nf} valid frames' in capt.out

            # run with a specific range, with negative values on end
            step(inp, use_frames='5:-5', options=options)
            capt = capsys.readouterr()
            assert f'{nf - 10} valid frames' in capt.out
            assert f'Removing {10} frames outside range'
