# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.instruments.flitecam.gaincor import gaincor
from sofia_redux.instruments.flitecam.tests.resources import intermediate_data


class TestGaincor(object):

    def test_success(self):
        hdul = intermediate_data()
        flux = hdul['FLUX'].data.copy()
        err = hdul['ERROR'].data.copy()

        # add a flat and flat error extension
        flat = np.full_like(hdul['FLUX'].data, 0.8)
        flaterr = np.full_like(hdul['FLUX'].data, 0.1)
        hdr = fits.Header({'FLATNORM': 100.0,
                           'ASSC_OBS': 'test'})
        hdul.append(fits.ImageHDU(flat, hdr, name='FLAT'))
        hdul.append(fits.ImageHDU(flaterr, name='FLAT_ERROR'))

        # add an exposure and badmask to make sure they're propagated
        hdul.append(fits.ImageHDU(np.arange(10), name='EXPOSURE'))
        hdul.append(fits.ImageHDU(np.arange(10), name='BADMASK'))

        result = gaincor(hdul)

        # flux is corrected
        assert np.allclose(result['FLUX'].data, flux / flat, equal_nan=True)

        # error is propagated
        var = err ** 2 / flat ** 2 + flaterr ** 2 * flux ** 2 / flat ** 4
        assert np.allclose(result['ERROR'].data, np.sqrt(var), equal_nan=True)

        # flat extensions removed
        assert 'FLAT' not in result
        assert 'FLAT_ERROR' not in result

        # exposure and badmask kept
        assert 'EXPOSURE' in result
        assert 'BADMASK' in result

        # headers updated
        assert result[0].header['FLATNORM'] == 100
        assert 'test' in result[0].header['FLAT_OBS']

    def test_no_flat_error(self):
        hdul = intermediate_data()
        flux = hdul['FLUX'].data.copy()
        err = hdul['ERROR'].data.copy()

        # add a flat and flat error extension
        flat = np.full_like(hdul['FLUX'].data, 0.8)
        hdr = fits.Header({'FLATNORM': 100.0,
                           'ASSC_OBS': 'test'})
        hdul.append(fits.ImageHDU(flat, hdr, name='FLAT'))

        result = gaincor(hdul)

        # flux is corrected
        assert np.allclose(result['FLUX'].data, flux / flat, equal_nan=True)

        # error is propagated
        var = err ** 2 / flat ** 2
        assert np.allclose(result['ERROR'].data, np.sqrt(var), equal_nan=True)
