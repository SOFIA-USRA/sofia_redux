# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

import sofia_redux.instruments.flitecam as fdrp
import sofia_redux.instruments.flitecam.lincor as u
from sofia_redux.instruments.flitecam.tests.resources import raw_testdata


@pytest.fixture(scope='function')
def linfile():
    pth = os.path.join(os.path.dirname(fdrp.__file__),
                       'data', 'linearity_files')
    linfile = os.path.join(pth, 'flitecam_lc_coeffs.fits')
    return linfile


class TestLincor(object):

    def test_success(self, linfile):
        hdul = raw_testdata()
        flux = hdul[0].data.copy()

        result = u.lincor(hdul, linfile)

        # flux is corrected
        assert not np.allclose(result[0].data, flux)

        # error and badmask are added
        assert 'ERROR' in result
        assert 'BADMASK' in result

        # without saturation level provided, badmask is all good
        assert np.all(result['BADMASK'].data) == 0

    def test_saturation(self, linfile):
        # with saturation specified, badmask marks high flux
        hdul = raw_testdata(clean=True)
        flux = hdul[0].data.copy()
        medval = np.nanmedian(flux / hdul[0].header['DIVISOR'])

        result = u.lincor(hdul, linfile, saturation=medval)

        # about half the data should be marked above the median
        assert np.allclose(np.sum(result['BADMASK'].data), 0.5 * flux.size,
                           rtol=.01)

    def test_badfile(self, tmpdir, capsys):
        # missing file
        hdul = raw_testdata()
        with pytest.raises(ValueError) as err:
            u.lincor(hdul, 'badfile.fits')
        assert 'Missing linearity file' in str(err)

        # badly formatted file
        badfile = tmpdir.join('badfile.fits')
        badfile.write('bad')
        with pytest.raises(ValueError) as err:
            u.lincor(hdul, str(badfile))
        assert 'Missing linearity file' in str(err)
        assert 'Could not read FITS data' in capsys.readouterr().err

        # wrong shape data file
        lin = fits.HDUList(fits.PrimaryHDU(np.arange(10)))
        lin.writeto(str(badfile), overwrite=True)
        with pytest.raises(ValueError) as err:
            u.lincor(hdul, str(badfile))
        assert 'Linearity file has wrong shape' in str(err)

    def test_imgpoly_vector(self):
        # vector
        x = np.arange(10, dtype=float)

        # constant only
        coeffs = np.full((1, 10), 1)
        result = u._imgpoly(x, coeffs)
        assert result.shape == x.shape
        assert np.allclose(result, 1)

        # 3rd order, all 1s
        coeffs = np.full((4, 10), 1)
        result = u._imgpoly(x, coeffs)
        assert result.shape == x.shape
        assert np.allclose(result, x**3 + x**2 + x + 1)

        # 3rd order, non-1 coefficients
        coeffs = np.tile([.1, .2, .3, .4], 10).reshape(10, 4).T
        result = u._imgpoly(x, coeffs)
        assert result.shape == x.shape
        assert np.allclose(result, 0.4 * x**3 + 0.3 * x**2 + 0.2 * x + 0.1)

    def test_imgpoly_image(self):
        # image
        x = np.arange(100, dtype=float).reshape(10, 10)

        # constant only
        coeffs = np.full((1, 10, 10), 1)
        result = u._imgpoly(x, coeffs)
        assert result.shape == x.shape
        assert np.allclose(result, 1)

        # 3rd order, all 1s
        coeffs = np.full((4, 10, 10), 1)
        result = u._imgpoly(x, coeffs)
        assert result.shape == x.shape
        assert np.allclose(result, x**3 + x**2 + x + 1)

        # 3rd order, non-1 coefficients
        coeffs = np.tile([.1, .2, .3, .4], 100).reshape(100, 4).T
        coeffs = coeffs.reshape(4, 10, 10)
        result = u._imgpoly(x, coeffs)
        assert result.shape == x.shape
        assert np.allclose(result, 0.4 * x**3 + 0.3 * x**2 + 0.2 * x + 0.1)

    def test_linearize(self):
        x = np.arange(16, dtype=float).reshape(4, 4) + 10
        coeffs = np.full((4, x.shape[0], x.shape[1]), 1)
        itime = 1 / 2
        readtime = 1
        ndr = 1

        # no-op smoke test: all coefficients are null, ie.
        # all c0 / _imgpoly < 1
        result = u._linearize(x.copy(), coeffs, itime, readtime, ndr)
        assert np.allclose(result, x)

        # for other values
        coeffs = np.full((2, x.shape[0], x.shape[1]), -10.)
        coeffs[1] = .001
        result = u._linearize(x.copy(), coeffs, itime, readtime, ndr)

        # expected values from IDL are just slightly higher,
        # with higher values corrected more
        # from:
        #   IDL> mc_flitecamlincor(image, itime, readtime, ndr, coeff)
        expected = [[10.032119, 11.038665, 12.045817, 13.053580],
                    [14.061950, 15.070935, 16.080534, 17.09070],
                    [18.101564, 19.113005, 20.125057, 21.137728],
                    [22.151018, 23.164915, 24.179447, 25.194582]]
        assert np.allclose(result, expected)
