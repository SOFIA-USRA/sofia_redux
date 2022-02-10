# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepflat import StepFlat
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data


class TestFlat(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)

        # raw data fails
        step = StepFlat()
        with pytest.raises(KeyError):
            step(inp)

        # demod data passes
        out = step(dmd)
        assert isinstance(out, DataFits)

    def test_checksize(self, capsys):
        step = StepFlat()

        # fails
        datashape = (10, 20)
        flatshape = (20, 10)
        with pytest.raises(ValueError):
            step.checksize(datashape, flatshape)
        capt = capsys.readouterr()
        assert 'Flat does not fit data' in capt.err

        # extra frames in data okay
        datashape = (10, 10, 20)
        flatshape = (10, 20)
        step.checksize(datashape, flatshape)

        # extra frames in both okay
        datashape = (10, 10, 20)
        flatshape = (10, 10, 20)
        step.checksize(datashape, flatshape)

        # extra frames, but mismatched 2D not okay
        datashape = (10, 10, 30)
        flatshape = (10, 20)
        with pytest.raises(ValueError):
            step.checksize(datashape, flatshape)
        capt = capsys.readouterr()
        assert 'Flat does not fit data' in capt.err

        # too many dimensions in flat data
        datashape = (10, 20)
        flatshape = (10, 10, 20)
        with pytest.raises(ValueError):
            step.checksize(datashape, flatshape)
        capt = capsys.readouterr()
        assert 'Flat does not fit data' in capt.err

    def test_labmode(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)

        step = StepFlat()
        out = step(dmd, labmode=True)
        assert 'Lab flat (no correction)' in str(out.header['HISTORY'])
