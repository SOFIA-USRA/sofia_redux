# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from astropy.modeling.models import Gaussian2D
import numpy as np

import sofia_redux.instruments.flitecam.maskbp as u
from sofia_redux.instruments.flitecam.tests.resources import raw_testdata


class TestMaskbp(object):

    def test_success(self):
        hdul = raw_testdata(nx=650, ny=650)
        flux = hdul[0].data.copy()

        # add expected extensions
        hdul[0].header['EXTNAME'] = 'FLUX'
        hdul.append(fits.ImageHDU(flux * 0.1, name='ERROR'))
        hdul.append(fits.ImageHDU(np.full(flux.shape, 0), name='BADMASK'))

        result = u.maskbp(hdul)

        # flux is not corrected by default, but bad pixels are marked -
        # should be nearly all the 1500 hot + cold pixels added to
        # simulated data
        assert np.allclose(result[0].data, flux)
        assert np.allclose(np.sum(result['BADMASK'].data), 1500, atol=5)

        # with cval specified, bad pixels are corrected
        result = u.maskbp(hdul, cval=np.nan)
        assert np.all(np.isnan(result['FLUX'].data)
                      == (result['BADMASK'].data == 1))
        assert np.nanstd(result['FLUX'].data) < np.std(flux)

    def test_stamp_check(self):
        # flat stamp should be good
        stamp = np.full((5, 5), 10.0)
        assert u._test_stamp(stamp.copy(), 1, 0) == (False, 10.0)

        # flat stamp + hot pixel should be bad,
        # should return local median
        stamp[2, 2] = 1000
        assert u._test_stamp(stamp.copy(), 1, 0) == (True, 10.0)

        # same for cold pixel
        stamp *= -1
        assert u._test_stamp(stamp.copy(), 1, 0, sign=-1) == (True, -10.0)

        # gaussian stamp should be good
        gp = {'amplitude': 10.0, 'x_stddev': 1.0,
              'y_stddev': 1.0, 'x_mean': 2, 'y_mean': 2}
        y, x = np.mgrid[:5, :5]
        g = Gaussian2D(**gp)
        stamp = g(x, y)
        assert u._test_stamp(stamp.copy(), 1, 0) == (False, 10.0)

        # gaussian stamp + hot pixel should be bad,
        # should return local median
        stamp[2, 2] = 1000
        bad, val = u._test_stamp(stamp.copy(), 1, 0)
        assert bad
        assert np.allclose(val, 4.8720505)

        # same for cold pixel
        stamp[2, 2] = -1000
        bad, val = u._test_stamp(stamp.copy(), 1, 0, sign=-1)
        assert bad
        assert np.allclose(val, 4.8720505)
