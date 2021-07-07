# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from scipy.signal import convolve
import warnings

__all__ = ['smoothres']


def smoothres(x, y, resolution, siglim=5):
    """
    Smooth a data to a constant resolution

    The procedure is:

    1. Resample data to a constant spacing in log(wavelength).
    2. Convolve resampled data with Gaussian kernel.
    3. Interpolate back to linear wavelength spacing.

    Parameters
    ----------
    x : array_like of (int or float)
        (N,) independent variable
    y : array_like if (int or float)
        (N,) dependent variable
    resolution : int or float
        Spectral resolution to smooth to
    siglim : int or float, optional
        Maximum fwhm deviation

    Returns
    -------
    numpy.ndarray
        (N,) The smoothed data array.
    """
    x, y = np.array(x), np.array(y)
    if x.shape != y.shape:
        raise ValueError("x and y array shape mismatch")
    elif x.ndim != 1:
        raise ValueError("x and y arrays must have 1 dimension")
    elif resolution < 0:
        raise ValueError("resolution must be positive")
    elif np.allclose(resolution, 0):
        return y

    n = x.size
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        a = (n - 1) / (np.log(xmax) - np.log(xmin))
        b = n - (a * np.log(xmax))
        xpon = np.arange(n, dtype=float) + 1

        xlog = np.exp((xpon - b) / a)
        ylog = np.interp(xlog, x, y)

        # Resample to constant spacing in log lambda; dv/c = d(ln lambda)
        sigma = 1 / (resolution * 2 * np.sqrt(2 * np.log(2)))
        wgauss = (np.arange(n) - (n / 2)) / a
        idx = np.abs(wgauss / sigma) <= siglim

    nok = idx.sum()
    if nok > (n / 2):
        log.warning(
            "Kernel too large; "
            "only part of the input array will be correctly convolved")
    elif nok < 2:
        log.error("No data less than sigma limit: all data=%s" % np.nan)
        return np.full(y.shape, np.nan)

    psf = np.exp(-0.5 * ((wgauss[idx] / sigma) ** 2))
    psf /= psf.sum()

    # perform convolution
    yconv = convolve(ylog, psf, mode='same')
    yconv[~np.isfinite(yconv)] = 0

    # Switch back to linear x-spacing
    yout = np.interp(x, xlog, yconv)
    return yout
