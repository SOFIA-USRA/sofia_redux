# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepscanpolmerge \
    import StepScanPolMerge
from sofia_redux.instruments.hawc.steps.stepscanpolmerge \
    import Reduction as SMR
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, scan_raw_data # , scan_sim_data


@pytest.fixture(scope='function')
def test_options():
    # options for faster test reductions
    return 'rounds=1'
    # return ''


@pytest.mark.timeout(0)
class TestScanPolMerge(DRPTestCase):

    def make_scanpol_data(self, tmpdir, good=True, nframe=20):
        if good:
            angle = [5.0, 50.0, 27.0, 72.0]
        else:
            angle = [5.0, 50.0]
        inp = []
        for i in range(len(angle)):
            hdul = scan_raw_data(nframe=nframe)
            # hdul = scan_sim_data(nframe=nframe)

            hdul[2].data['hwpCounts'] = angle[i] * 4

            ffile = str(tmpdir.join('test{}.fits'.format(i)))
            hdul.writeto(ffile, overwrite=True)
            hdul.close()

            inp.append(DataFits(ffile))
        return inp

    def test_miso(self, tmpdir, test_options):
        inp = self.make_scanpol_data(tmpdir)

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            step = StepScanPolMerge()

            out = step(inp, options=test_options)
            assert isinstance(out, DataFits)

    def test_bad_file(self, tmpdir, capsys, test_options):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        hdul.close()
        df = DataFits(ffile)

        # also test badly formatted options: they should be ignored
        opt = test_options + ' q'

        with tmpdir.as_cwd():
            step = StepScanPolMerge()
            with pytest.raises(ValueError):
                step([df], options=opt)
            capt = capsys.readouterr()
            assert 'No scans to reduce' in capt.err

    def test_scan_errors(self, tmpdir, capsys, mocker, test_options):
        with tmpdir.as_cwd():
            # incorrect number of angles
            inp = self.make_scanpol_data(tmpdir, good=False)
            step = StepScanPolMerge()
            with pytest.raises(ValueError) as err:
                step(inp, options=test_options)
            assert 'Files do not consist of 4 HWP groups' in str(err)

            # bad return type
            inp = self.make_scanpol_data(tmpdir, good=True)
            mocker.patch.object(SMR, 'run', return_value=inp)
            with pytest.raises(ValueError) as err:
                step(inp, options=test_options)
            assert 'Expected output not found' in str(err)

    def test_scanpol(self, tmpdir, capsys, test_options):
        # make scan pol data
        inp = self.make_scanpol_data(tmpdir)
        exp = 0
        for df in inp:
            exp += df.header['EXPTIME']

        # move to tmpdir -- writes temp files
        with tmpdir.as_cwd():
            step = StepScanPolMerge()
            out = step(inp, options=test_options)
            names = ['STOKES I', 'ERROR I',
                     'STOKES Q', 'ERROR Q',
                     'STOKES U', 'ERROR U',
                     'COVAR Q I', 'COVAR U I', 'COVAR Q U',
                     'BAD PIXEL MASK']
            for name in names:
                assert name in out.imgnames

            # exposure time in primary header should sum over input
            #assert np.allclose(out.getheadval('EXPTIME'), exp)
            assert out.getheadval('PRODTYPE') == 'scanpolmerge'
            assert 'SPR' in out.filename

            # check that if only 1 angle is passed, it is processed
            # as an imaging file instead
            out = step([inp[0]], options=test_options)
            assert isinstance(out, DataFits)
            capt = capsys.readouterr()
            assert 'Found 1 HWP; processing as imaging data' in capt.err
            assert out.getheadval('PRODTYPE') == 'scanmap'
            assert 'SMP' in out.filename

    def test_smap_options(self, tmpdir, mocker, capsys):
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

        step = StepScanPolMerge()

        # default options --
        # no output saved, so will fail
        with pytest.raises(ValueError):
            step([df])
        capt = capsys.readouterr()
        assert 'No output created' in capt.err
        # options are not in scanmap command string
        assert 'deep=True' not in capt.out
        assert 'faint=True' not in capt.out
        assert 'extended=True' not in capt.out
        # df.filename is in command
        assert os.path.basename(fname) in capt.out

        # deep option
        with pytest.raises(ValueError):
            step([df], deep=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'deep=True' in capt.out

        # faint option
        with pytest.raises(ValueError):
            step([df], faint=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'faint=True' in capt.out

        # extended option
        with pytest.raises(ValueError):
            step([df], extended=True)
        step.arglist = {}
        capt = capsys.readouterr()
        assert 'extended=True' in capt.out

        # if df.filename is not on disk, it will check for
        # df.rawname
        df.rawname = df.filename
        df.filename = 'badfile.fits'
        with pytest.raises(ValueError):
            step([df])
        step.arglist = {}
        capt = capsys.readouterr()
        assert os.path.basename(fname) in capt.out
        assert 'badfile.fits' not in capt.out

    def test_scanmappol_frame_clip(self, tmpdir, capsys):
        """Test frame clipping options, in scanpol mode."""
        # make some data to test
        nf = 80
        inp = self.make_scanpol_data(tmpdir, nframe=nf)

        with tmpdir.as_cwd():
            step = StepScanPolMerge()

            # run with all frames used, turning off some other
            # clipping options
            options = 'downsample=1 blacklist=vclip,fillgaps ' \
                      'shift=0 chopper.shift=0 rounds=1'
            step(inp, use_frames='', options=options)
            capt = capsys.readouterr()
            assert f'Reading {nf} frames' in capt.out
            assert f'{nf} valid frames' in capt.out

            # clip some from beginning and end
            step(inp, use_frames='5:-5', options=options)
            capt = capsys.readouterr()
            assert f'{nf - 10} valid frames' in capt.out
            assert f'Removing {10} frames outside range'

    def test_grid(self, tmpdir, capsys, test_options):
        """Test grid option."""
        inp = self.make_scanpol_data(tmpdir)
        with tmpdir.as_cwd():
            step = StepScanPolMerge()

            # run with no grid: uses default
            out = step(inp, options=test_options)
            capt = capsys.readouterr()
            assert 'Grid Spacing: 3.4 x 3.4 arcsec' in capt.out
            assert np.isclose(out.header['CDELT1'], -3.4 / 3600)
            assert np.isclose(out.header['CDELT2'], 3.4 / 3600)

            # specify grid
            out = step(inp, grid=6.8, options=test_options)
            capt = capsys.readouterr()
            assert 'Grid Spacing: 6.8 x 6.8 arcsec' in capt.out
            assert np.isclose(out.header['CDELT1'], -6.8 / 3600)
            assert np.isclose(out.header['CDELT2'], 6.8 / 3600)
