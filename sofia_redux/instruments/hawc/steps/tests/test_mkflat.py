# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepmkflat import StepMkflat
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, intcal_raw_data


class TestMkFlat(DRPTestCase):
    def make_data(self, tmpdir, nfiles=2):
        inp = []
        for i in range(nfiles):
            hdul = intcal_raw_data()
            ffile = str(tmpdir.join('testRAW{}.fits'.format(i)))
            hdul.writeto(ffile, overwrite=True)
            df = DataFits(ffile)

            # assign a config to test filename generation
            df.config['data']['filenum'] = r'.*(\d+)\.fits'
            df.config['data']['filenamebegin'] = '.*test'
            df.config['data']['filenameend'] = r'\d+.fits'

            # prepare and demodulate
            prep = StepPrepare()(df)
            dmd = StepDemodulate()(prep)
            inp.append(dmd)
        return inp

    def test_mimo(self, tmpdir):
        # test nominal default run
        inp = self.make_data(tmpdir)

        # drops flats directory in cwd
        with tmpdir.as_cwd():
            step = StepMkflat()
            out = step(inp)
            assert isinstance(out, list)
            assert len(out) == len(inp)
            assert isinstance(out[0], DataFits)
            assert os.path.isdir('flats')
            assert os.path.isfile('flats/testOFT0-1.fits')

    def test_mkflat_errors(self, tmpdir, capsys, mocker):
        inp = self.make_data(tmpdir)
        with tmpdir.as_cwd():
            step = StepMkflat()

            # make a flat folder to use
            os.makedirs('testflats', exist_ok=True)

            # mock makedirs to trigger os error
            def mock_makedirs(*args, **kwargs):
                raise OSError('test error')
            mocker.patch('os.makedirs', mock_makedirs)

            with pytest.raises(OSError):
                step(inp, flatoutfolder='badfolder')
            capt = capsys.readouterr()
            assert 'Failed to make flatoutfolder' in capt.err

            # if folder already exists, run will succeed
            step(inp, flatoutfolder='testflats')
            assert os.path.isfile('testflats/testOFT0-1.fits')
            capsys.readouterr()

            # if no folder, will write to current directory
            step(inp, flatoutfolder='')
            assert os.path.isfile('testOFT0-1.fits')
            capsys.readouterr()

            # try to run on bad flat
            step = StepMkflat()
            inp2 = []
            for df in inp:
                df2 = df.copy()
                df2.table['R Array'] *= np.nan
                df2.table['T Array'] *= np.nan
                inp2.append(df2)
            with pytest.raises(ValueError):
                step(inp2, flatoutfolder='testflats')
            capt = capsys.readouterr()
            assert 'No good flat files found' in capt.err

    def test_groups(self, tmpdir):
        inp = self.make_data(tmpdir, nfiles=4)
        with tmpdir.as_cwd():
            step = StepMkflat()

            # change scriptid so there are two groups instead of 1
            inp[2].setheadval('SCRIPTID', '98765')
            inp[3].setheadval('SCRIPTID', '98765')
            step(inp)
            assert os.path.isfile('flats/testOFT0-1.fits')
            assert os.path.isfile('flats/testOFT2-3.fits')

    def test_dcl_only(self, tmpdir):
        inp = self.make_data(tmpdir)
        with tmpdir.as_cwd():
            step = StepMkflat()
            # set option to write dcl files, rather than flats
            out = step(inp, dcl_only=True)
            assert os.path.isdir('flats')
            assert not os.path.isfile('flats/testOFT0-1.fits')
            assert os.path.isfile('flats/testDCL0.fits')
            assert os.path.isfile('flats/testDCL1.fits')
            assert len(out) == 0
