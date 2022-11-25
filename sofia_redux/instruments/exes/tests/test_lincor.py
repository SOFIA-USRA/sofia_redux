# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import os
import pytest

from sofia_redux.instruments.exes import lincor


@pytest.fixture
def header():
    datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'data')
    darkfile = os.path.join(datadir, 'dark', 'dark_2015.02.13.fits')
    linfile = os.path.join(datadir, 'lincoeff', 'EXES_nlcoefs_7_20150703.fits')
    header = fits.Header()
    header['NSPAT'] = 1024
    header['NSPEC'] = 1024
    header['DRKFILE'] = darkfile
    header['LINFILE'] = linfile
    return header


class TestLinCor(object):

    def test_check_data(self):

        # Only 2 or 3 dimensions are allowed
        with pytest.raises(ValueError):
            lincor._check_data(np.zeros(10))
            lincor._check_data(np.zeros((2, 2, 2, 2)))

        assert lincor._check_data(np.empty((10, 10))).shape == (1, 10, 10)
        assert lincor._check_data(np.empty((10, 10, 10))).shape == (10, 10, 10)

    def test_get_linearity_coefficients(self, header):

        linfile = header['LINFILE']
        nx = header['NSPAT']
        ny = header['NSPEC']

        header['LINFILE'] = '__does_not_exist__'
        with pytest.raises(ValueError) as err:
            lincor._get_linearity_coefficients(header)
        assert "could not read linearity file" in str(err).lower()

        header['NSPAT'] = nx + 1
        header['NSPEC'] = ny + 1
        header['LINFILE'] = linfile
        with pytest.raises(ValueError) as err:
            lincor._get_linearity_coefficients(header)
        assert 'too small for data' in str(err).lower()

        header['NSPAT'] = nx
        header['NSPEC'] = ny
        minframe, maxframe, coeffs = lincor._get_linearity_coefficients(header)
        assert minframe.shape == (ny, nx)
        assert maxframe.shape == (ny, nx)
        assert coeffs.shape[1:] == (ny, nx)
        assert np.issubdtype(np.float, coeffs.dtype)
        assert np.issubdtype(np.int, minframe.dtype)
        assert np.issubdtype(np.int, maxframe.dtype)

    def test_apply_correction(self, header):
        bias = lincor.get_reset_dark(header)
        minframe, maxframe, coeffs = lincor._get_linearity_coefficients(header)
        bias[...] = 1000  # apply flat bias
        minframe[...] = 0
        maxframe[...] = 10

        data = np.full((2, bias.shape[0], bias.shape[1]), 998.0)

        # Create fake coefficients that are easy to check
        coeffs = np.empty((3, bias.shape[0], bias.shape[1]))
        coeffs[0] = 0.0
        coeffs[1] = 0.2
        coeffs[2] = 0.1
        result, mask = lincor._apply_correction(
            data, header, bias, minframe, maxframe, coeffs)

        # 1000 - (2 / (0.1 * 2 ** 2 + 0.2 * 2))
        assert np.allclose(result, 997.5)
        assert mask.all()

        data[:, :data.shape[1] // 2] -= 1000
        result, mask = lincor._apply_correction(
            data, header, bias, minframe, maxframe, coeffs)
        assert np.allclose(np.unique(result), [-2, 997.5])
        assert not mask.all() and mask.any()

    def test_lincor(self, header):
        bias = lincor.get_reset_dark(header)
        data = bias[None].astype(float)
        result, mask = lincor.lincor(data, header)
        assert np.issubdtype(float, result.dtype)
        assert np.issubdtype(bool, mask.dtype)
        assert result.shape == data.shape
        assert mask.shape == data.shape
