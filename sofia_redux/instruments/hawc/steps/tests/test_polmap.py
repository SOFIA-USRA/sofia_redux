# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.steppolvec import StepPolVec
from sofia_redux.instruments.hawc.steps.stepregion import StepRegion
from sofia_redux.instruments.hawc.steps.steppolmap import StepPolMap
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestPolMap(DRPTestCase):
    def make_data(self, tmpdir, name='test.fits'):
        # note -- should cd to tmpdir before calling this.
        hdul = pol_bgs_data()

        # add some necessary keywords from other steps
        hdul[0].header['BMIN'] = 0.003778
        hdul[0].header['BMAJ'] = 0.003778
        hdul[0].header['BPA'] = 0.0

        ffile = str(tmpdir.join(name))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        vec = StepPolVec()(df)

        # set parameters to preserve vectors
        reg = StepRegion()(vec, minisigi=0, sigma=0)

        return reg

    def test_siso(self, tmpdir):
        with tmpdir.as_cwd():
            df = self.make_data(tmpdir)
            step = StepPolMap()

            # passes on polvec data with defaults
            out = step(df)
            assert isinstance(out, DataFits)
            assert os.path.isfile('test.PMP_polmap.png')

    def test_run_options(self, tmpdir, capsys):
        with tmpdir.as_cwd():
            df = self.make_data(tmpdir)
            step = StepPolMap()

            # output file name
            fname = 'test.PMP_polmap.png'

            # run with nhwp -- just returns
            df1 = df.copy()
            df1.setheadval('NHWP', 1)
            step(df1)
            capt = capsys.readouterr()
            assert 'No polarization data, so skipping step' in capt.out
            assert not os.path.isfile(fname)

            # plot data with no vectors -- still produces image
            df1 = df.copy()
            df1.tabledel('FINAL POL DATA')
            step(df1)
            capt = capsys.readouterr()
            assert 'No vectors found' in capt.out
            assert os.path.isfile(fname)
            os.remove(fname)

            # use center cropping
            par = [df.getheadval('CRVAL1'),
                   df.getheadval('CRVAL2'),
                   10 * df.getheadval('CDELT1'),
                   10 * df.getheadval('CDELT1')]
            step(df, centercrop=True, centercropparams=par)
            capt = capsys.readouterr()
            assert 'Using center cropping' in capt.out

            # run with/without rotation
            step(df, rotate=True)
            capt = capsys.readouterr()
            assert 'Plotting B vectors' in capt.out
            step(df, rotate=False)
            capt = capsys.readouterr()
            assert 'Plotting E vectors' in capt.out
            os.remove(fname)

            # more options: un-debiased data, scale flag,
            # fill contours, more vectors
            step(df, debias=False, scale=False, fillcontours=False)
            assert os.path.isfile(fname)

            # scale low/high parameter
            # bad input
            with pytest.raises(TypeError):
                step(df, lowhighscale="['a', 'b']")
            # okay input
            step(df, lowhighscale=[0, 100])

    def test_threadsafe(self, tmpdir, capsys):
        # set log to error level to ignore warnings
        from astropy import log
        log.setLevel('ERROR')

        with tmpdir.as_cwd():

            def _try_plot(i):
                df = self.make_data(tmpdir, name=f'test{i}.fits')
                step = StepPolMap()
                step(df)

            # this will crash with a fatal Python error
            # if plots are not thread safe

            from threading import Thread
            t1 = Thread(target=_try_plot, args=(1,))
            t1.setDaemon(True)
            t1.start()
            t2 = Thread(target=_try_plot, args=(2,))
            t2.setDaemon(True)
            t2.start()

            # let both finish
            t1.join()
            t2.join()

            # check for output
            assert os.path.exists('test1.PMP_polmap.png')
            assert os.path.exists('test2.PMP_polmap.png')
