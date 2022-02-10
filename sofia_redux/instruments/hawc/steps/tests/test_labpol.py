# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.steppolvec import StepPolVec
from sofia_redux.instruments.hawc.steps.stepregion import StepRegion
from sofia_redux.instruments.hawc.steps.steplabpolplots import StepLabPolPlots
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestLabPol(DRPTestCase):
    def make_data(self, tmpdir, suffix=''):
        hdul = pol_bgs_data()
        # move to tmpdir -- writes auxfiles to cwd
        with tmpdir.as_cwd():
            ffile = str(tmpdir.join(f'test{suffix}.fits'))
            hdul.writeto(ffile, overwrite=True)
            df = DataFits(ffile)
            vec = StepPolVec()(df)
            reg = StepRegion()(vec)
        return df, reg

    def test_siso(self, tmpdir):
        df, reg = self.make_data(tmpdir)

        # move to tmpdir -- writes auxfiles to cwd
        with tmpdir.as_cwd():
            # fails on non-vector data
            step = StepLabPolPlots()
            with pytest.raises(ValueError):
                step(df)

            # passes on polvec data
            out = step(reg)
            assert isinstance(out, DataFits)

    def test_threadsafe(self, tmpdir, capsys):
        with tmpdir.as_cwd():

            def _try_plot(i):
                df, reg = self.make_data(tmpdir, suffix=i)
                step = StepLabPolPlots()
                step(reg)

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

            # check for output from both threads
            fname = ['test1.PLT.png', 'test2.PLT.png']
            for fn in fname:
                assert os.path.exists(fn)
