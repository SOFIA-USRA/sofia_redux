# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import units as u
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepdmdplot import StepDmdPlot
from sofia_redux.instruments.hawc.tests.resources import \
    DRPTestCase, pol_raw_data, intcal_raw_data, add_col, del_col


class TestDmdPlot(DRPTestCase):
    def make_data(self, tmpdir, name='test.fits'):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join(name))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)
        return dmd

    def test_science_file(self, tmpdir, capsys):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)

        # cd to tmpdir -- this function drops png files
        with tmpdir.as_cwd():
            # raw data fails
            step = StepDmdPlot()
            with pytest.raises(KeyError):
                step(inp)

            # demod data passes
            out = step(dmd)
            assert isinstance(out, DataFits)
            assert os.path.isfile('test.DPL.png')

            # various tracking conditions

            hdul1 = del_col(hdul.copy(), 'TrackErrAoi3')
            inp.table = hdul1[2].data
            prep = StepPrepare()(inp)
            dmd = StepDemodulate()(prep)
            step(dmd)
            capt = capsys.readouterr()
            assert 'Found no AOI keys' in capt.out

            hdul1 = add_col(hdul.copy(), 'CentroidAoi', 'TrackErrAoi3')
            inp.table = hdul1[2].data
            prep = StepPrepare()(inp)
            dmd = StepDemodulate()(prep)
            step(dmd)
            capt = capsys.readouterr()
            assert 'Found AOI key: CentroidAoi' in capt.out

            hdul1 = add_col(hdul.copy(), 'SofHkTrkaoi', 'TrackErrAoi3')
            inp.table = hdul1[2].data
            prep = StepPrepare()(inp)
            dmd = StepDemodulate()(prep)
            step(dmd)
            capt = capsys.readouterr()
            assert 'Found AOI key: SofHkTrkaoi' in capt.out

            # no centroid plot, data_iters=0
            hdul1 = del_col(hdul.copy(), 'CentroidExpMsec')
            inp.table = hdul1[2].data
            prep = StepPrepare()(inp)
            dmd = StepDemodulate()(prep)
            step(dmd, data_iters=0)
            capt = capsys.readouterr()
            assert 'CentroidExpMsec not found' in capt.out

            # set a save folder
            os.makedirs('test', exist_ok=True)
            savefolder = str(tmpdir.join('test'))
            step(dmd, savefolder=savefolder)
            assert os.path.isfile(os.path.join(savefolder, 'test.DPL.png'))

    def test_intcal(self, tmpdir, capsys):
        hdul = intcal_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)

        with tmpdir.as_cwd():
            step = StepDmdPlot()

            # first run on a file without calmode -- unknown is assumed
            unk = dmd.copy()
            del unk.header['CALMODE']
            step(unk)
            assert os.path.isfile('test.DPL.png')
            capt = capsys.readouterr()
            assert 'median R phase' not in capt.out
            assert 'median T phase' not in capt.out

            # now run on intcal
            step(dmd)
            assert os.path.isfile('test.DPL.png')
            capt = capsys.readouterr()
            assert 'median R phase offset =' in capt.out
            assert 'median T phase offset =' in capt.out

            # also save phase
            step(dmd, save_phase=True)
            capsys.readouterr()
            assert os.path.isfile('test.PHS.fits')
            phase = fits.open('test.PHS.fits')
            assert 'RPHASE' in phase
            assert 'TPHASE' in phase
            assert phase[0].header['INSTRUME'] == 'HAWC_PLUS'
            assert phase[0].header['PRODTYPE'] == 'phaseoffset'
            assert phase[0].header['PROCSTAT'] == 'LEVEL_2'
            phase.close()

            # now with no auxphase, no sigma clipping
            step(dmd, ref_phase_file='', data_iters=0)
            capt = capsys.readouterr()
            assert os.path.isfile('test.DPL.png')
            assert 'median R phase =' in capt.out
            assert 'median T phase =' in capt.out

            # various tracking conditions

            hdul1 = del_col(hdul.copy(), 'TrackErrAoi3')
            inp.table = hdul1[2].data
            prep = StepPrepare()(inp)
            dmd = StepDemodulate()(prep)
            step(dmd)
            capt = capsys.readouterr()
            assert 'Found no AOI keys' in capt.out

            hdul1 = add_col(hdul.copy(), 'CentroidAoi', 'TrackErrAoi3')
            inp.table = hdul1[2].data
            prep = StepPrepare()(inp)
            dmd = StepDemodulate()(prep)
            step(dmd)
            capt = capsys.readouterr()
            assert 'Found AOI key: CentroidAoi' in capt.out

            hdul1 = add_col(hdul.copy(), 'SofHkTrkaoi', 'TrackErrAoi3')
            inp.table = hdul1[2].data
            prep = StepPrepare()(inp)
            dmd = StepDemodulate()(prep)
            step(dmd)
            capt = capsys.readouterr()
            assert 'Found AOI key: SofHkTrkaoi' in capt.out

    def test_calc_phase(self):
        chop_freq = 1.0
        user_freq = 1.0
        step = StepDmdPlot()

        # real and imag equal => phase is -45
        real = np.full(10, 1.0)
        imag = np.full(10, 1.0)
        phase = step.calc_phase(real, imag, chop_freq, user_freq)
        assert np.allclose(phase.value, -45.)

        # imag zero => phase is 0
        real = np.full(10, 1.0)
        imag = np.full(10, 0.0)
        phase = step.calc_phase(real, imag, chop_freq, user_freq)
        assert np.allclose(phase.value, 0.)

        # real zero => phase is -90
        real = np.full(10, 0.0)
        imag = np.full(10, 1.0)
        phase = step.calc_phase(real, imag, chop_freq, user_freq)
        assert np.allclose(phase.value, -90.)

        # with reference=expected, phase returned is zero
        real = np.full(10, 1.0)
        imag = np.full(10, 1.0)
        ref = np.full(10, -45.) * u.deg
        phase = step.calc_phase(real, imag, chop_freq, user_freq,
                                phaseref=ref)
        assert np.allclose(phase.value, 0.)

    def test_threadsafe(self, tmpdir, capsys):
        with tmpdir.as_cwd():

            df1 = self.make_data(tmpdir, name='test1.fits')
            df2 = self.make_data(tmpdir, name='test2.fits')
            inp = [df1, df2]

            def _try_plot(i):
                step = StepDmdPlot()
                step(inp[i])

            # this will crash with a fatal Python error
            # if plots are not thread safe

            from threading import Thread
            t1 = Thread(target=_try_plot, args=(0,))
            t1.setDaemon(True)
            t1.start()
            t2 = Thread(target=_try_plot, args=(1,))
            t2.setDaemon(True)
            t2.start()

            # let both finish
            t1.join()
            t2.join()

            # check for output
            assert os.path.exists('test1.DPL.png')
            assert os.path.exists('test2.DPL.png')
