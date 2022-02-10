# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.steppolvec import StepPolVec
from sofia_redux.instruments.hawc.steps.stepregion import StepRegion
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestRegion(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        vec = StepPolVec()(df)

        # move to temp dir -- drops region files
        with tmpdir.as_cwd():
            # fails on non-vector data
            step = StepRegion()
            with pytest.raises(RuntimeError):
                step(df)

            # passes on polvec data
            out = step(vec)
            assert isinstance(out, DataFits)

    def test_no_hwp(self, capsys):
        df = DataFits()
        df.setheadval('NHWP', 1)
        step = StepRegion()
        # no-op with nhwp=1
        out = step(df)
        capt = capsys.readouterr()
        assert 'Only 1 HWP, so skipping step' in capt.out
        assert 'FINAL POL DATA' not in out.tabnames

    def test_reg_options(self, tmpdir):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        vec = StepPolVec()(df)

        with tmpdir.as_cwd():
            step = StepRegion()

            # run with fewer cuts to preserve vectors
            out = step(vec, mini=0.0001, minp=0.0001, minisigi=0, sigma=0)
            tab = out.tableget('FINAL POL DATA')
            assert os.path.isfile('test.REG.reg')
            assert 'FINAL POL DATA' in out.tabnames

            # use unrotated, unscaled -- affects vector plot only
            out1 = step(vec, mini=0.0001, minp=0.0001, minisigi=0, sigma=0,
                        rotate=False, scale=False)
            tab1 = out1.tableget('FINAL POL DATA')
            for col in tab1.names:
                assert np.allclose(tab1[col], tab[col])

            # use undebiased -- should let more vectors through
            out1 = step(vec, mini=0.0001, minp=0.0001, minisigi=0, sigma=0,
                        debias=False)
            tab1 = out1.tableget('FINAL POL DATA')
            assert len(tab1) > len(tab)
