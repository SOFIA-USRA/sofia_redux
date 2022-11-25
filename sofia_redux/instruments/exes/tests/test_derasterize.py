# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.exes import derasterize as dr
from sofia_redux.instruments.exes.readhdr import readhdr


class TestDerasterize(object):

    def make_data(self, nstripe=4, overlap=32):
        nx = 1032
        ny = 1024 // nstripe + overlap
        nz = nstripe
        flat = np.full((nz, ny, nx), -2.0)
        dark = np.full((nz, ny, nx), -1.0)

        # vary the stripes
        flat *= np.arange(nz)[:, None, None] + 1

        # set simple OTPAT
        header = fits.Header()
        header['OTPAT'] = 'D0'
        header['NINT'] = 1
        header['DATE-OBS'] = '2015-01-01T00:00:00.000'
        header['INSTCFG'] = 'HIGH_LOW'
        header = readhdr(header, check_header=False)

        return flat, dark, header

    @pytest.mark.parametrize('nstripe,overlap', [(2, 32), (4, 32), (8, 32),
                                                 (2, 16), (4, 16), (8, 16)])
    def test_derasterize(self, nstripe, overlap):
        flat, dark, header = self.make_data(nstripe=nstripe, overlap=overlap)

        dflat, var, mask = dr.derasterize(flat, header, overlap=overlap,
                                          dark_data=dark, dark_header=header)

        assert dflat.shape == (1, 1024, 1024)
        assert var.shape == (1, 1024, 1024)
        assert mask.shape == (1024, 1024)

        # flat should have even values, with stripes corrected to
        # first value and corrected for the number of y pixels
        ny = 1024 // nstripe + overlap
        expected = 1024 / ny
        nnan = ~np.isnan(dflat[0, :-2])
        assert np.allclose(dflat[0, :-2][nnan], expected)
        # except the top row, which is always 0
        assert np.all(dflat[0, -1, :] == 0)

        # without the separate dark, the reset dark is in the output flat,
        # so with a different zero point and a little more variable
        expected *= 10000
        dflat, var, mask = dr.derasterize(flat, header, overlap=overlap)
        mask[-2:, :] = False
        assert np.allclose(dflat[0][mask], expected, rtol=0.2)
        # except the top row, which is always 0
        assert np.all(dflat[:, -1, :] == 0)

    def test_derasterize_errors(self):
        flat, dark, header = self.make_data()

        with pytest.raises(RuntimeError) as err:
            dr.derasterize(flat, header, dark_data=np.arange(10))
        assert 'Dark data has wrong dimensions' in str(err)

        with pytest.raises(RuntimeError) as err:
            dr.derasterize(flat, header, dark_data=dark)
        assert 'Dark header must be provided' in str(err)

        with pytest.raises(RuntimeError) as err:
            dr.derasterize(flat, header, overlap=16)
        assert 'overlap of 16 rows does not match data' in str(err)

        with pytest.raises(RuntimeError) as err:
            dr.derasterize(flat[:2], header)
        assert 'Number of stripes (4) does not match data' in str(err)
