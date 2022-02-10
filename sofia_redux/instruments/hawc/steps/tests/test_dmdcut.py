# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepdmdcut import StepDmdCut
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data


class TestDmdCut(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)

        # raw data fails
        step = StepDmdCut()
        with pytest.raises(KeyError):
            step(inp)

        # demod data passes
        out = step(dmd)
        assert isinstance(out, DataFits)
