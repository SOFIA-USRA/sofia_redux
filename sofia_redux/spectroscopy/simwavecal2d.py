# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.fitting.polynomial import poly1d
import warnings

__all__ = ['simwavecal2d']


def simwavecal2d(shape, edgecoeffs, xranges, slith_arc, ds):
    """
    Simulate a 2D wavecal file using pixels for wavelengths

    NOTE: will also return indices for new iSHELL wavecal FITS
    format.

    Parameters
    ----------
    shape : 2-tuple of int
        (nrow, ncol) shape of image.
    edgecoeffs : array_like of float
        (norders, 2, degree+1) array of polynomial coefficients which
        define the edges of the orders. edgecoeffs[0, 0, :] are the
        coefficients of the bottom edge of the first order and
        edgecoeffs[0, 1, :] are the coefficients of the top edge of the
        first order.
    xranges : array_like of float
        (norders, 2) array of column numbers between which the orders
        are completely on the array.
    slith_arc : float
        The slit height in arcseconds.
    ds : float
        The plate scale in arcseconds per pixel.

    Returns
    -------
    wavecal, spatcal, indices : numpy.ndarray, numpy.ndarray, dict
        - wavecal (nrow, ncol) array where each pixel is set to its
          wavelength (column in this case).
        - spatcal (nrow, ncol) array where each pixel is set to its
          angular position on the sky.
        - indices (dict of dict) where 1st keys are orders (int) and
          second level keys are as follows:

              ``"x"``
                  x indices of constant wavelength and angle in
                  the zeroth order.
              ``"y"``
                  y indices of constant wavelength and angle in
                  the zeroth order.
              ``"xgrid"``
                  (ncol) array of wavelength values along x
              ``"ygrid"``
                  (nrow) array of spatial coordinates along y

    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        log.error("Invalid shape")
        return
    edgecoeffs = np.array(edgecoeffs)
    if edgecoeffs.ndim != 3 or edgecoeffs.shape[1] != 2:
        log.error("Invalid edgecoeffs shape")
        return
    norders = edgecoeffs.shape[0]
    xranges = np.array(xranges)
    if xranges.shape != (norders, 2):
        log.error("Invalid xranges shape")
        return
    wavecal = np.full(shape, np.nan)
    spatcal = np.full(shape, np.nan)
    indices = {}

    nsgrid = int(np.round(slith_arc / ds)) + 1
    sgrid = np.arange(nsgrid) * ds

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(norders):
            minx, maxx = xranges[i, 0], xranges[i, 1]
            nx = int(maxx - minx) + 1
            ny = shape[0]
            x = np.arange(nx) + minx
            y = np.arange(ny)
            pixtoarc = np.zeros((nx, 2))

            # Find the bottom and top of the slit
            botedge = poly1d(x, edgecoeffs[i, 0, :])
            topedge = poly1d(x, edgecoeffs[i, 1, :])
            pixtoarc[:, 1] = slith_arc / (topedge - botedge)
            pixtoarc[:, 0] = -pixtoarc[:, 1] * botedge

            # make sure bottom and top don't spill over array
            botedge[botedge < 0] = 0
            topedge[topedge >= ny] = ny - 1

            # Start indices
            ix = np.empty((nsgrid, nx))
            ix[None] = x.copy()
            iy = np.full((nsgrid, nx), np.nan)

            for j in range(nx):
                j0, j1 = int(np.floor(botedge[j])), int(np.ceil(topedge[j]))
                wavecal[j0: j1, int(x[j])] = x[j]
                ypix = y[j0: j1]
                spix = poly1d(ypix, pixtoarc[j])
                spatcal[j0: j1, int(x[j])] = spix.copy()
                iy[:, j] = np.interp(sgrid, spix, ypix)

            indices[i] = {'x': ix, 'y': iy,
                          'xgrid': x.copy(),
                          'ygrid': sgrid.copy()}

    return wavecal, spatcal, indices
