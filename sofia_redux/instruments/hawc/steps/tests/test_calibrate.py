# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepcalibrate import StepCalibrate
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestCalibrate(DRPTestCase):
    def test_siso(self, tmpdir):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepCalibrate()
        out = step(inp)

        assert isinstance(out, DataFits)

    def test_errors(self, capsys):
        # run on empty data
        df = DataFits()
        df.imageset(np.zeros((10, 10)), imagename='PRIMARY IMAGE')
        df.imageset(np.zeros((10, 10)), imagename='NOISE')

        step = StepCalibrate()
        step.datain = df

        # passes, but data is uncalibrated,
        # since it's missing keywords
        step.run()
        capt = capsys.readouterr()
        assert 'No calibration factor found; ' \
               'not calibrating data' in capt.err

        # try to run on a standard: just returns
        df.setheadval('OBSTYPE', 'STANDARD_FLUX')
        step.run()
        capt = capsys.readouterr()
        assert 'Flux standard; not applying calibration' in capt.out

        # run with 1 HWP - should still run, with error
        df = DataFits()
        df.imageset(np.zeros((10, 10)), imagename='STOKES I')
        df.imageset(np.zeros((10, 10)), imagename='ERROR I')
        df.setheadval('NHWP', 1)
        step.datain = df
        step.run()
        capt = capsys.readouterr()
        assert 'No calibration factor found; ' \
               'not calibrating data' in capt.err
