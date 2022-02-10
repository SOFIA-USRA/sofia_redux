# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps import basehawc
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, basic_raw_data


class TestBaseHawc(DRPTestCase):
    def test_readfunc(self, tmpdir):
        """
        Test all the read functions in basehawc.

        All tested together to avoid redoing set-up steps.
        """
        # test data -- increase frames for this test case
        hdul = basic_raw_data(nframe=320, smplfreq=1)
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)

        # prepare to call basehawc
        step = StepDemodulate()
        step.datain = prep
        step.runstart(prep, {})
        step.readdata()

        # test readchop

        # chop state is potentially made of 0, 1, -1 -- this
        # data set should be evenly split between 1 and -1
        chopstate = basehawc.readchop(step, step.praw['nsamp'], 5.0,
                                      chopflag=True)
        zero = chopstate[chopstate == 0]
        high = chopstate[chopstate == 1]
        low = chopstate[chopstate == -1]
        assert zero.size + high.size + low.size == chopstate.size
        assert high.size == low.size
        assert zero.size == 0

        # when chopflag is False, should be only ones
        chopstate = basehawc.readchop(step, step.praw['nsamp'], 5.0,
                                      chopflag=False)
        assert np.all(chopstate == 1)

        # test readnod

        # nod state is just like chop state, but
        # zeros between nod values for these data
        nodstate = basehawc.readnod(step, step.praw['nsamp'], 5.0,
                                    nodflag=True)
        zero = nodstate[nodstate == 0]
        high = nodstate[nodstate == 1]
        low = nodstate[nodstate == -1]
        assert zero.size + high.size + low.size == nodstate.size
        assert high.size == low.size
        assert zero.size == 16

        # when nodflag is False, should be only ones
        nodstate = basehawc.readnod(step, step.praw['nsamp'], 5.0,
                                    nodflag=False)
        assert np.all(nodstate == 1)

        # test readhwp
        # angles in test data are: 5.0 50.0 27.0 72.0, ~80 samples each

        # hwp state for these data should be mostly 1s, a few 0s
        # (3 at the beginning + 3, between angles);
        # nhwp should be 4
        hwpstate, nhwp = basehawc.readhwp(step, step.praw['nsamp'], 5.0, 10.2)
        assert nhwp == 4
        zero = hwpstate[hwpstate == 0]
        high = hwpstate[hwpstate == 1]
        assert zero.size + high.size == hwpstate.size
        assert zero.size == 6

        # set the tolerance too low -- all invalid points
        hwpstate, nhwp = basehawc.readhwp(step, step.praw['nsamp'], 0, 10.2)
        assert np.all(hwpstate == 0)

        # set a bad data region in the middle of the first angle --
        # will trim data to start after bad region
        step.praw['HWP Angle'][7:20] = 50.0
        hwpstate, nhwp = basehawc.readhwp(step, step.praw['nsamp'], 5.0, 10.2)
        assert np.where(hwpstate == 1)[0][0] == 21
        # hwp index set to -1 for invalid, 0-3 for hwp angles 1-4
        assert step.praw['HWP Index'][20] == -1
        assert step.praw['HWP Index'][21] == 0
        assert step.praw['HWP Index'][-1] == 3

        # set a few bad samples (> tol) at the beginnings of hwp regions
        # should be trimmed to exclude

        # end of first angle
        step.praw['HWP Angle'][78] = 6.5
        step.praw['HWP Angle'][79] = 7.5
        # beginning of third angle
        step.praw['HWP Angle'][160:180] = 24.5
        step.praw['HWP Angle'][180] = 25.5
        hwpstate, nhwp = basehawc.readhwp(step, step.praw['nsamp'], 2.0, 10.2)
        assert np.all(hwpstate[78:80]) == 0
        assert np.all(hwpstate[160:181] == 0)
        assert np.where(step.praw['HWP Index'] == 2)[0][0] == 181
