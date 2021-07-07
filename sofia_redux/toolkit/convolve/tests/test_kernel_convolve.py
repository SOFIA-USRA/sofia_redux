# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.modeling.models import Gaussian2D
from astropy.stats.funcs import gaussian_fwhm_to_sigma
import numpy as np

from sofia_redux.toolkit.convolve.kernel import KernelConvolve, convolve


def test_expected():
    y, x, = np.mgrid[:100, :100]
    sdev = gaussian_fwhm_to_sigma * 2
    model = Gaussian2D(amplitude=1.0, x_mean=50, y_mean=50,
                       x_stddev=sdev, y_stddev=sdev)
    kernel = model(x, y)

    image = np.zeros_like(kernel)
    image[30, 30] = 2.0

    result = KernelConvolve(x, y, image, kernel, error=image + 1,
                            normalize=False, do_error=True)

    imax = np.argwhere(result.result == result.result.max())
    assert np.allclose(imax, 30)
    assert np.isclose(result.result[30, 30], 2.0)
    assert np.isclose(result.result[31, 30], 1.0)
    assert np.isclose(result.error[30, 30], 2.0672497536610357)
    assert np.isclose(result.error[30, 31], 1.6653893070425347)

    # also test convolve wrapper for this case
    wimg, werr = convolve(x, y, image, kernel, error=image + 1,
                          normalize=False, do_error=None)
    assert np.allclose(result.result, wimg)
    assert np.allclose(result.error, werr)

    # no error, with do_error=False
    wimg = convolve(x, y, image, kernel, error=image + 1,
                    normalize=False, do_error=False)
    assert np.allclose(result.result, wimg)

    # no error, with do_error=True, error=None
    wimg = convolve(x, y, image, kernel, error=None,
                    normalize=False, do_error=True)
    assert np.allclose(result.result, wimg)

    # with stats
    wimg, werr, wstats = convolve(x, y, image, kernel, error=image + 1,
                                  normalize=False, do_error=True,
                                  stats=True)
    assert np.allclose(result.result, wimg)
    assert np.allclose(result.error, werr)
    assert np.allclose(result.stats.chi2, wstats.chi2)
