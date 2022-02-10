# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepprepare import StepPrepare
from sofia_redux.instruments.hawc.steps.stepdemodulate import StepDemodulate
from sofia_redux.instruments.hawc.steps.stepflat import StepFlat
from sofia_redux.instruments.hawc.steps.stepsplit import StepSplit
from sofia_redux.instruments.hawc.steps.stepcombine import StepCombine
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data, pol_raw_data


class TestCombine(DRPTestCase):
    def make_pre_comb(self, tmpdir, mode='pol'):
        """Minimal steps to make data that will pass combination step."""
        hdul = pol_raw_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)

        steps = [StepPrepare, StepDemodulate, StepFlat, StepSplit]
        inp = DataFits(ffile)
        for step in steps:
            s = step()
            inp = s(inp)

        hdul = inp.to_hdulist()
        if mode != 'pol':
            new_hdul = fits.HDUList()
            for ext in hdul:
                if 'HWP0' not in ext.header['EXTNAME']:
                    continue
                else:
                    new_hdul.append(ext)
            hdul = new_hdul
            hdul[0].header['NHWP'] = 1

        hdul.writeto(ffile, overwrite=True)
        return ffile

    def name_vals(self, df, covar=False, nhwp=4):
        names = []
        for hwp in range(nhwp):
            for nod in range(2):
                names.append('DATA R-T HWP%d NOD%d' % (hwp, nod))
                names.append('DATA R+T HWP%d NOD%d' % (hwp, nod))
                names.append('VAR R HWP%d NOD%d' % (hwp, nod))
                names.append('VAR T HWP%d NOD%d' % (hwp, nod))
                names.append('ERROR R-T HWP%d NOD%d' % (hwp, nod))
                names.append('ERROR R+T HWP%d NOD%d' % (hwp, nod))
                names.append('COVAR R-T HWP%d NOD%d' % (hwp, nod))
                names.append('COVAR R+T HWP%d NOD%d' % (hwp, nod))
        for name in names:
            if 'COVAR' in name and not covar:
                assert name not in df.imgnames
            else:
                assert name in df.imgnames
                # data should be combined in all images,
                # so no more cubes after this step
                assert df.imageget(name).ndim == 2

    def test_siso(self, tmpdir):
        ffile = self.make_pre_comb(tmpdir)
        inp = DataFits(ffile)

        step = StepCombine()
        out = step(inp)
        assert isinstance(out, DataFits)
        self.name_vals(out)

    def test_badfile(self, tmpdir, capsys):
        hdul = pol_bgs_data()
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        inp = DataFits(ffile)

        step = StepCombine()
        with pytest.raises(ValueError):
            step(inp)
        capt = capsys.readouterr()
        assert 'invalid image name' in capt.err

    def test_use_error(self, tmpdir, capsys):
        ffile = self.make_pre_comb(tmpdir)
        inp = DataFits(ffile)
        step = StepCombine()
        step.datain = inp

        step.runstart(inp, {'use_error': True})
        step.run()
        chv_err = np.nanmean(step.dataout.imageget('ERROR R-T HWP0 NOD0'))
        capt = capsys.readouterr()
        assert 'Covariances between initial Stokes parameters ' \
               'are not propagated with Chauvenet errors' in capt.err

        step.runstart(inp, {'use_error': False})
        step.run()
        prop_err = np.nanmean(step.dataout.imageget('ERROR R-T HWP0 NOD0'))

        # should give different answers
        assert not np.allclose(chv_err, prop_err)

    def test_non_pol(self, tmpdir, capsys):
        ffile = self.make_pre_comb(tmpdir, mode='imaging')
        inp = DataFits(ffile)

        step = StepCombine()
        out = step(inp)
        assert isinstance(out, DataFits)
        self.name_vals(out, nhwp=1)
