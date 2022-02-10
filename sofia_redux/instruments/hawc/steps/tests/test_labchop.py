# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.steplabchop import StepLabChop
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_raw_data


class TestLabChop(DRPTestCase):
    def make_data(self, tmpdir, name='test.fits'):
        hdul = pol_raw_data()
        ffile = str(tmpdir.join(name))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)
        prep = StepPrepare()(inp)
        dmd = StepDemodulate()(prep)
        return dmd

    def test_run(self, tmpdir):
        df = self.make_data(tmpdir)
        step = StepLabChop()

        # passes on demod data
        out = step(df)
        assert isinstance(out, DataFits)

        # should contain modulus and phase images, no configuration image
        assert 'MODULUS' in out.imgnames
        assert 'PHASE' in out.imgnames
        assert 'CONFIGURATION' not in out.imgnames

        # phase angles should all be between 0 and 360
        phase = out.imageget('PHASE')
        nn = ~np.isnan(phase)
        assert np.all(phase[nn] >= 0)
        assert np.all(phase[nn] < 360)

        # both images should have shape 41 x 128
        modulus = out.imageget('MODULUS')
        assert phase.shape == (41, 128)
        assert modulus.shape == (41, 128)

        # run again, with 2nd T detector zeroed out: output images are cut
        df.table['T array'][:, :, 32:] = 0
        df.table['T array Imag'][:, :, 32:] = 0
        out = step(df)
        phase = out.imageget('PHASE')
        modulus = out.imageget('MODULUS')
        assert phase.shape == (41, 96)
        assert modulus.shape == (41, 96)
