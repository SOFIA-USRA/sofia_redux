# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepflat import StepFlat
from sofia_redux.instruments.hawc.steps.stepsplit import StepSplit
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, pol_raw_data


class TestSplit(DRPTestCase):
    def make_pre_split(self, tmpdir):
        """Minimal steps to make data that will pass split step."""
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
        ffile = self.make_pre_split(tmpdir)
        inp = DataFits(ffile)

        step = StepSplit()
        out = step(inp)
        assert isinstance(out, DataFits)

    def test_badfile(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepSplit()
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'invalid image name' in capt.err

    def test_chop_error(self, tmpdir, capsys):
        ffile = self.make_pre_split(tmpdir)
        df = DataFits(ffile)
        step = StepSplit()

        # 0 chops in 1st left
        df1 = df.copy()
        df1.table['HWP Index'] = 1
        with pytest.raises(ValueError):
            step(df1)
        capt = capsys.readouterr()
        assert 'Zero chops in 1st left' in capt.err

        # 0 chops in 2nd left
        df1 = df.copy()
        df1.table['Nod Index'] = 0
        with pytest.raises(ValueError):
            step(df1)
        capt = capsys.readouterr()
        assert 'Zero chops in 2nd left' in capt.err

        # 0 chops in right
        df1 = df.copy()
        tab = df1.table['Nod Index']
        df1.table['Nod Index'][tab == 1] = 2
        with pytest.raises(ValueError):
            step(df1)
        capt = capsys.readouterr()
        assert 'Zero chops in right' in capt.err

        # difference in chop numbers > tolerance -- warns
        step(df, nod_tol=-1)
        capt = capsys.readouterr()
        assert 'Number of chops between 1st and 2nd left differ' in capt.err
        assert 'Number of chops between left and right differ' in capt.err

    def test_nodpatt(self, tmpdir, capsys):
        ffile = self.make_pre_split(tmpdir)
        df = DataFits(ffile)
        step = StepSplit()

        # A only -- 1 img per hwp (4) * 6 + badpix
        # (data/var for R+T, R-T, var for R, T)
        df.setheadval('NODPATT', 'A')
        out = step(df)
        assert len(out.imgnames) == 25

        # ABBA -- 2 img per hwp (4) * 6
        df.setheadval('NODPATT', 'ABBA')
        out = step(df)
        assert len(out.imgnames) == 49

        # any other A/B nodpatt -> error
        df.setheadval('NODPATT', 'AB')
        with pytest.raises(ValueError):
            step(df)
        capt = capsys.readouterr()
        assert 'Can only process data with ABBA nod pattern' in capt.err

    def test_rtarray(self, tmpdir):
        ffile = self.make_pre_split(tmpdir)
        df = DataFits(ffile)
        step = StepSplit()

        # output sets unused array variance to 0
        test_r = 'VAR R HWP0 NOD0'
        test_t = 'VAR T HWP0 NOD0'

        # default -- both arrays
        out = step(df.copy(), rtarrays='RT')
        assert not np.allclose(out.imageget(test_r), 0)
        assert not np.allclose(out.imageget(test_t), 0)

        # R only -- T is set to 0
        out = step(df.copy(), rtarrays='R')
        assert not np.allclose(out.imageget(test_r), 0)
        assert np.allclose(out.imageget(test_t), 0)

        # T only -- R is set to 0
        out = step(df.copy(), rtarrays='T')
        assert np.allclose(out.imageget(test_r), 0)
        assert not np.allclose(out.imageget(test_t), 0)
