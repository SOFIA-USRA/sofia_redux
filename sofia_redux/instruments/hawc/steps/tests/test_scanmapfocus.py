# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.stepmiparent import StepMIParent
from sofia_redux.instruments.hawc.steps.stepscanmapfocus \
    import StepScanMapFocus
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, scan_raw_data


class LogInput(StepMIParent):
    """Minimal step for testing."""
    def run(self):
        for df in self.datain:
            log.info(df.filename)
        self.dataout = self.datain[0].copy()


class TestScanMapFocus(DRPTestCase):
    def test_mimo(self, tmpdir):
        inp = []
        for i in range(2):
            hdul = scan_raw_data()
            ffile = str(tmpdir.join('test{}.fits'.format(i)))
            hdul[0].header['FILENAME'] = os.path.basename(ffile)
            hdul.writeto(ffile, overwrite=True)
            inp.append(DataFits(ffile))

        # will reduce all files together, since they match
        # (uses scanmap, with default config)
        step = StepScanMapFocus()
        out = step(inp, options='rounds=1')

        # output is single element list
        assert isinstance(out, list)
        assert len(out) == 1
        for df in out:
            assert isinstance(df, DataFits)

    def test_grouping(self, capsys, mocker):
        # minimal test data
        inp = []
        for i in range(8):
            df = DataFits()
            df.imageset(np.zeros((10, 10)))
            # two groups -- even and odd
            df.setheadval('TESTKEY', i % 2)
            df.filename = 'test{}.fits'.format(i)
            df.setheadval('FILENAME', df.filename)
            inp.append(df)

        # mock scanmap
        mocker.patch(
            'sofia_redux.instruments.hawc.steps.stepscanmapfocus.StepScanMap',
            LogInput)

        kwargs = {'groupkeys': 'TESTKEY',
                  'groupkfmt': ''}

        step = StepScanMapFocus()
        out = step(inp, **kwargs)
        assert len(out) == 2
        # evens
        assert 'test0' in out[0].filename
        # odds
        assert 'test1' in out[1].filename
        capt = capsys.readouterr()
        for df in inp:
            assert df.filename in capt.out
