# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepflat import StepFlat
from sofia_redux.instruments.hawc.steps.stepsplit import StepSplit
from sofia_redux.instruments.hawc.steps.stepcombine import StepCombine
from sofia_redux.instruments.hawc.steps.stepnodpolsub import StepNodPolSub
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, pol_raw_data


class TestNodPolSub(DRPTestCase):
    def make_pre_nps(self, tmpdir):
        """Minimal steps to make data that will pass nodpolsub step."""
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        steps = [StepPrepare, StepDemodulate, StepFlat,
                 StepSplit, StepCombine]
        inp = DataFits(ffile)
        for step in steps:
            s = step()
            inp = s(inp)

        hdul = inp.to_hdulist()
        hdul.writeto(ffile, overwrite=True)
        return ffile

    def test_siso(self, tmpdir):
        ffile = self.make_pre_nps(tmpdir)
        inp = DataFits(ffile)

        step = StepNodPolSub()
        out = step(inp)
        assert isinstance(out, DataFits)
        assert 'DATA R+T HWP0' in out.imgnames
        assert 'DATA R-T HWP0' in out.imgnames

        # also test chop-nod data (nhwp=1)
        inp.setheadval('NHWP', 1)
        step = StepNodPolSub()
        out = step(inp)
        assert 'DATA R+T HWP0' in out.imgnames
        assert 'DATA R-T HWP0' not in out.imgnames

    def test_badfile(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepNodPolSub()
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'invalid image name' in capt.err

    def test_nodpatt(self, tmpdir, capsys):
        ffile = self.make_pre_nps(tmpdir)
        inp = DataFits(ffile)
        step = StepNodPolSub()

        # set nod pattern to A only -- does no subtraction
        inp.setheadval('NODPATT', 'A')

        step(inp)
        capt = capsys.readouterr()
        assert 'No nod subtraction' in capt.out

        # set nod pattern to something invalid
        inp.setheadval('NODPATT', 'ABBAAB')
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'Can only process data with ABBA nod pattern' in capt.err
