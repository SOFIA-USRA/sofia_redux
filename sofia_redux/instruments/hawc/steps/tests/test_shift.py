# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepflat import StepFlat
from sofia_redux.instruments.hawc.steps.stepshift import StepShift
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, pol_raw_data


class TestShift(DRPTestCase):
    def make_pre_shift(self, tmpdir):
        """Minimal steps to make data that will pass shift step."""
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        steps = [StepPrepare, StepDemodulate, StepFlat]
        inp = DataFits(ffile)
        for step in steps:
            s = step()
            inp = s(inp)

        hdul = inp.to_hdulist()
        hdul.writeto(ffile, overwrite=True)
        return ffile

    def test_siso(self, tmpdir):
        ffile = self.make_pre_shift(tmpdir)
        inp = DataFits(ffile)

        step = StepShift()
        out = step(inp)
        assert isinstance(out, DataFits)

    def test_badfile(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepShift()
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'No valid tables' in capt.err

    def test_bad_track(self, capsys):
        df = DataFits()
        step = StepShift()

        # fails if TRCKSTAT is bad
        df.setheadval('TRCKSTAT', "BAD TRACKING")
        with pytest.raises(ValueError):
            step(df)
        capt = capsys.readouterr()
        assert 'Bad file due to tracking issues' in capt.err

    def test_nonzero_shift(self, tmpdir, capsys):
        ffile = self.make_pre_shift(tmpdir)
        df = DataFits(ffile)
        step = StepShift()

        # default -- zero shift
        out = step(df, disp1=[0, 0], disp2=[0, 0])
        capt = capsys.readouterr()
        assert 'Shifted R array' not in capt.out
        r_im = out.imageget('R Array')

        # apply 1 pix shift in y for R1
        out1 = step(df, disp1=[0, 1], disp2=[0, 1])
        capt = capsys.readouterr()
        assert 'Shifted R array' in capt.out
        r_im1 = out1.imageget('R Array')
        assert np.allclose(r_im[:, :-1, :], r_im1[:, 1:, :],
                           equal_nan=True)
