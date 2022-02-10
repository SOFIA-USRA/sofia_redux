# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.instruments.hawc.datafits import DataFits
from sofia_redux.instruments.hawc.steps.stepbinpixels import StepBinPixels
from sofia_redux.instruments.hawc.tests.resources \
    import DRPTestCase, pol_bgs_data


class TestBinPixels(DRPTestCase):
    def make_data(self, tmpdir, imaging=False):
        hdul = pol_bgs_data()
        if imaging:
            pol_ext = ['STOKES Q', 'STOKES U', 'COVAR Q I',
                       'COVAR U I', 'COVAR Q U']
            for ext in pol_ext:
                del hdul[ext]
        ffile = str(tmpdir.join('test.fits'))
        hdul.writeto(ffile, overwrite=True)
        df = DataFits(ffile)
        return df

    def test_siso(self, tmpdir):
        inp = self.make_data(tmpdir)
        step = StepBinPixels()
        out = step(inp)
        assert isinstance(out, DataFits)

    def test_no_bin(self, tmpdir):
        inp = self.make_data(tmpdir)
        step = StepBinPixels()
        out = step(inp, block_size=1)
        assert out.image.shape == inp.image.shape
        assert np.allclose(out.image, inp.image, equal_nan=True)

    def test_bad_bin(self, tmpdir):
        inp = self.make_data(tmpdir)
        step = StepBinPixels()
        with pytest.raises(ValueError) as err:
            step(inp, block_size=3)
        assert 'does not divide' in str(err)

    def test_good_bin(self, tmpdir):
        inp = self.make_data(tmpdir)
        step = StepBinPixels()
        # sizes 2, 4, 8 should work
        out2 = step(inp, block_size=2)
        assert out2.image.shape == tuple([s // 2 for s in inp.image.shape])
        out4 = step(inp, block_size=4)
        assert out4.image.shape == tuple([s // 4 for s in inp.image.shape])
        out8 = step(inp, block_size=8)
        assert out8.image.shape == tuple([s // 8 for s in inp.image.shape])

        # total flux should be conserved
        assert np.allclose(np.nansum(inp.image), np.nansum(out2.image))
        assert np.allclose(np.nansum(inp.image), np.nansum(out4.image))
        assert np.allclose(np.nansum(inp.image), np.nansum(out8.image))

        # all extensions should have the same shape
        shape2 = tuple([s // 2 for s in inp.image.shape])
        for img in out2.imgnames:
            assert out2.imageget(img).shape == shape2

        # bad pixel mask should have values 0-3
        assert np.all(out2.imageget('BAD PIXEL MASK') >= 0)
        assert np.all(out2.imageget('BAD PIXEL MASK') <= 3)

    def test_imaging(self, tmpdir):
        inp = self.make_data(tmpdir, imaging=True)
        step = StepBinPixels()
        # binning still works, ignoring missing stokes extensions
        out2 = step(inp, block_size=2)
        assert out2.image.shape == tuple([s // 2 for s in inp.image.shape])
