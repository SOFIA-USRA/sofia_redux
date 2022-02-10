# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepflat import StepFlat
from sofia_redux.instruments.hawc.steps.stepsplit import StepSplit
from sofia_redux.instruments.hawc.steps.stepcombine import StepCombine
from sofia_redux.instruments.hawc.steps.stepnodpolsub import StepNodPolSub
from sofia_redux.instruments.hawc.steps.stepstokes import StepStokes
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, pol_raw_data


class TestStokes(DRPTestCase):
    def make_pre_stokes(self, tmpdir):
        """Minimal steps to make data that will pass stokes step."""
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        steps = [StepPrepare, StepDemodulate, StepFlat,
                 StepSplit, StepCombine, StepNodPolSub]
        inp = DataFits(ffile)
        for step in steps:
            s = step()
            inp = s(inp)

        hdul = inp.to_hdulist()
        hdul.writeto(ffile, overwrite=True)
        return ffile

    def test_siso(self, tmpdir, capsys):
        ffile = self.make_pre_stokes(tmpdir)
        inp = DataFits(ffile)

        step = StepStokes()
        out = step(inp)
        assert isinstance(out, DataFits)

        capt = capsys.readouterr()
        assert 'Stokes Q: HWP angles differ' not in capt.err
        assert 'Stokes U: HWP angles differ' not in capt.err

    def test_badfile(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepStokes()
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'invalid image name' in capt.err

    def test_erri(self, tmpdir, capsys):
        ffile = self.make_pre_stokes(tmpdir)
        inp = DataFits(ffile)
        step = StepStokes()

        # add some noise to make things clearer
        baseval = 10
        for img in inp.imgnames:
            if 'ERROR R+T' in img:
                err = inp.imageget(img)
                inp.imageset(
                    err + np.random.normal(baseval, baseval * 2, err.shape),
                    imagename=img)

        # no error inflation, standard propagation
        out = step(inp, erri='none', erripolmethod='meansigma')
        outerr = np.nanmean(out.imageget('ERROR I'))

        # inflate by median
        out1 = step(inp, erri='median', erripolmethod='meansigma')
        outerr1 = np.nanmean(out1.imageget('ERROR I'))
        assert outerr1 > outerr

        # inflate by mean
        out2 = step(inp, erri='mean', erripolmethod='meansigma')
        outerr2 = np.nanmean(out2.imageget('ERROR I'))
        assert np.allclose(outerr2, outerr1, rtol=0.1)

        # bad inflation parameter
        with pytest.raises(ValueError):
            step(inp, erri='badval')
        capt = capsys.readouterr()
        assert 'must be MEDIAN, MEAN, or NONE' in capt.err

        # no inflation, use hwpstddev calculation
        out1 = step(inp, erri='none', erripolmethod='hwpstddev')
        outerr1 = np.nanmean(out1.imageget('ERROR I'))
        assert not np.allclose(outerr1, outerr, rtol=0.1,
                               atol=(outerr1 / 10))

        # bad propagation parameter
        with pytest.raises(ValueError):
            step(inp, erripolmethod='badval')
        capt = capsys.readouterr()
        assert 'must be HWPSTDDEV or MEANSIGMA' in capt.err

    def test_hwp_angles(self, tmpdir, capsys):
        ffile = self.make_pre_stokes(tmpdir)
        inp = DataFits(ffile)
        step = StepStokes()

        # 1 angle -- make stokes I only
        inp.setheadval('NHWP', 1)
        out = step(inp)
        assert 'STOKES I' in out.imgnames
        assert 'STOKES Q' not in out.imgnames

        # not multiple of 4 -- raises error
        inp.setheadval('NHWP', 3)
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'Number of HWP angles must be ' \
               'multiple of 4' in capt.err

        # more than 16 -- raises error
        inp.setheadval('NHWP', 20)
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'Maximum number of HWP angles is 16' in capt.err

        # modify angles so that they are no longer 45 degrees
        # apart, within tolerance -- should warn, but continue
        inp.setheadval('NHWP', 4)
        tab1 = inp.tableget('TABLE HWP0')
        tab2 = inp.tableget('TABLE HWP2')
        tab1['HWP Angle'] += 5.0
        tab2['HWP Angle'] += 5.0
        inp.tableset(tab1, 'TABLE HWP0')
        inp.tableset(tab2, 'TABLE HWP1')
        step(inp, hwp_tol=0)
        capt = capsys.readouterr()
        assert 'Stokes Q: HWP angles differ' in capt.err
        assert 'Stokes U: HWP angles differ' in capt.err

    def test_manual_hwp_override(self, tmpdir, capsys):
        ffile = self.make_pre_stokes(tmpdir)
        inp = DataFits(ffile)
        step = StepStokes()

        # normal angle sort order is 1, 3, 2, 4,
        # so that first two are combined for Q,
        # last two are combined for U

        # modify angle sequence so that the wrong sets are combined
        # (sort order 1, 2, 3, 4; Q will be 0 and 2, U 1 and 3)
        tab0 = inp.tableget('TABLE HWP0')
        tab1 = inp.tableget('TABLE HWP1')
        tab2 = inp.tableget('TABLE HWP2')
        tab3 = inp.tableget('TABLE HWP3')
        tab0['HWP Angle'] = 5.0
        tab1['HWP Angle'] = 25.0
        tab2['HWP Angle'] = 50.0
        tab3['HWP Angle'] = 70.0
        inp.tableset(tab0, 'TABLE HWP0')
        inp.tableset(tab1, 'TABLE HWP1')
        inp.tableset(tab2, 'TABLE HWP2')
        inp.tableset(tab3, 'TABLE HWP3')

        # will warn about unexpected index values
        step(inp)
        capt = capsys.readouterr()
        assert 'HWP indices are: 0, 2' in capt.out
        assert 'Unexpected indices for Stokes Q' in capt.err
        assert 'HWP indices are: 1, 3' in capt.out
        assert 'Unexpected indices for Stokes U' in capt.err

        # now use manual override option
        step(inp, override_hwp_order=True)
        capt = capsys.readouterr()

        # will warn about angle difference instead, but
        # will use the expected index order (0, 1; 2, 3)
        assert 'HWP indices are: 0, 1' in capt.out
        assert 'Stokes Q: HWP angles differ by 20.0 degrees' in capt.err
        assert 'HWP indices are: 2, 3' in capt.out
        assert 'Stokes U: HWP angles differ by 20.0 degrees' in capt.err
        assert 'Unexpected indices' not in capt.err
