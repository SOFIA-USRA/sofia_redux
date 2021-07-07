# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.fitting.polynomial import polyfitnd
from sofia_redux.toolkit.convolve.filter import sobel
from sofia_redux.toolkit.interpolate import tabinv
from sofia_redux.toolkit.utilities.func import nantrim
import warnings

__all__ = ['findorders']


def findorders(image, guesspos, sranges=None, step=5, slith_range=None,
               degree=4, frac=0.8, comwin=5, ybuffer=3):
    """
    Determines the position of the order(s) in a spectral image

    Parameters
    ----------
    image : array_like of float
        (nrow, ncol) array from which to find edge coefficients and xranges
        for each order.  Flat field image oriented so that the dispersion
        axis is roughly aligned with the rows.
    guesspos : array_like of float
        (norders, 2) array giving the (x, y) (col, row) estimate for the
        center of each order.  For example, guesspos[5, 1] gives the
        center y-position estimate for the 6th order.  The positions should
        be near the center of the order and be located in a region of large
        flux away from bad pixels.
    sranges : array_like of int, optional
        (norders, 2) array giving the (start, stop) columns for each order.
        If None is supplied, sranges is calculated via `nantrim` over the
        full x-range of the image.
    step : int, optional
        Step size in the dispersion (column or x) direction.
    slith_range : array_like of float, optional
        (2,) array giving the range of possible slit height in pixels. i.e.
        (minimum height, maximum height).  This is used to make sure the
        routine doesn't include bad pixels in the fit.
    degree : int, optional
        Polynomial fit degree for the edges of the orders.
    frac : float, optional
        The fraction of the flux of the center of the slit used to identify
        the location of the edge of the order.
    comwin : int, optional
        The Center-Of-Mass window giving the number of pixels that should be
        used to determine an edge position along each column (y-direction).
    ybuffer : int, optional
        Buffer in pixels around the edge of the image.  Anything inside the
        buffer is marked as being "off" the image.

    Returns
    -------
    edgecoeffs, xranges : 2-tuple of numpy.ndarray
        edgecoeffs (norders, 2, degree+1) contains the lower and upper edge
        polynomial coefficients for each order.  For example, the coefficients
        for the fifth order top edge are found at edgecoeffs[4, 1] while the
        bottom edge coefficients are at edgecoeffs[4, 0].  xranges (norders, 2)
        give the x-ranges for each order where the slit is fully on the image,
        much like `sranges`.
    """
    halfwin = comwin // 2
    window = np.array([-halfwin, halfwin + 1])[None]
    guesspos = np.asarray(guesspos, dtype=float)
    if guesspos.ndim == 1:
        guesspos = np.array([guesspos])
    if guesspos.shape[1] != 2:
        raise ValueError("guesspos must have 2 elements per order")
    norders = guesspos.shape[0]

    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2 dimensional")
    shape = image.shape

    if sranges is None:
        sranges = image.copy()
        sranges[sranges == 0] = np.nan
        sranges = nantrim(sranges, 2)
        sranges = np.any(sranges, axis=0)
        xlims = np.argmax(sranges), shape[1] - np.argmax(sranges[::-1])
        sranges = np.empty((norders, 2), dtype=int)
        sranges[:, 0] = xlims[0]
        sranges[:, 1] = xlims[1] - 1
    else:
        sranges = np.asarray(sranges, dtype=int)
        if sranges.ndim == 1:
            sranges = np.array([sranges])

    if sranges.shape[0] != norders:
        raise ValueError(
            "number of orders in sranges does not match guesspos")
    elif sranges.shape[1] != 2:
        raise ValueError(
            "sranges must have 2 elements per order")

    rows = np.arange(shape[0])
    maxrow = shape[0] - 1
    ylimit = shape[0] - ybuffer
    edgecoeffs = np.full((norders, 2, degree + 1), np.nan)
    xranges = np.zeros((norders, 2), dtype=int)

    # scale, and roberts the image
    rimage = sobel(image * 1000 / np.nanmax(image))
    increment = np.array([-1, 1])
    do_height_check = slith_range is not None

    for i in range(norders):
        start, stop = sranges[i]
        starts = start + step - 1
        stops = stop - step + 1
        scols = np.arange(int((stops - starts) / step) + 1) * step + starts
        fcols = image[:, scols]
        rcols = rimage[:, scols]
        nscols = scols.size
        edges = np.full((2, nscols), np.nan)
        cen = np.full(nscols, np.nan)

        # Set up array to store the position of the center of the order
        # once the edges are found
        gidx = int(np.round(tabinv(scols, guesspos[i, 0])))
        cen[(gidx - degree): (gidx + degree + 1)] = guesspos[i, 1]
        fit_order = np.clip(degree - 2, 1, None)

        center_model = polyfitnd(scols, cen, fit_order, model=True)
        offset = np.zeros(2, dtype=int)
        botleft = topleft = 0
        botright = topright = shape[1] - 1

        while True:

            idx = np.unique(gidx + offset)
            idx = idx[(idx >= 0) & (idx < nscols)]
            if idx.size == 0:
                break

            is_left = (idx <= gidx)
            dobot = (is_left & (idx >= botleft)) \
                | (~is_left & (idx <= botright))
            dotop = (is_left & (idx >= topleft)) \
                | (~is_left & (idx <= topright))

            if not dobot.any() and not dotop.any():
                break

            yguesses = np.round(center_model(scols.take(idx)))
            yguesses = np.clip(yguesses, 0, maxrow).astype(int)
            zguesses = fcols[np.round(yguesses).astype(int), idx]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                zthresh = \
                    np.take(fcols, idx, axis=1) <= (frac * zguesses[None])
            ztop = (rows[:, None] > yguesses[None]) & zthresh
            topidx = np.argmax(ztop, axis=0)
            zbot = (rows[:, None] < yguesses[None]) & zthresh
            botidx = ztop.shape[0] - 1 - np.argmax(zbot[::-1], axis=0)

            botidx = np.clip(botidx[:, None] + window, 0, shape[0])
            topidx = np.clip(topidx[:, None] + window, 0, shape[0])
            zbot = np.any(zbot, axis=0)
            ztop = np.any(ztop, axis=0)

            for j, col in enumerate(idx):
                cy0 = cy1 = np.nan
                if zbot[j] and dobot[j]:
                    y = rows[botidx[j, 0]:botidx[j, 1]]
                    z = rcols[botidx[j, 0]:botidx[j, 1], col]
                    cy0 = np.nansum(y * z) / np.nansum(z)

                if ztop[j] and dotop[j]:
                    y = rows[topidx[j, 0]:topidx[j, 1]]
                    z = rcols[topidx[j, 0]:topidx[j, 1], col]
                    cy1 = np.nansum(y * z) / np.nansum(z)

                # Check the slit height is reasonable
                if dotop[j] and dobot[j] and (
                        np.isfinite(cy0) and np.isfinite(cy1)):
                    dy = abs(cy1 - cy0)
                    if not do_height_check or ([0] <= dy <= slith_range[1]):
                        edges[:, col] = cy0, cy1
                        cen[col] = (cy0 + cy1) / 2
                    else:
                        continue  # pragma: no cover
                else:
                    cen[col] = yguesses[j]

                if not (ybuffer < cy0 < ylimit):
                    if is_left[j]:
                        botleft = col
                    else:
                        botright = col

                if not (ybuffer < cy1 < ylimit):
                    if is_left[j]:
                        topleft = col
                    else:
                        topright = col

            center_model.refit_data(cen)
            offset += increment

        x = np.arange(start, stop + 1)
        in_slit = np.full(x.size, True)
        for j, edge in enumerate(edges):
            edge_model = polyfitnd(scols, edge, degree, robust=3, model=True)
            edgecoeffs[i, j] = edge_model.coefficients
            if np.isnan(edgecoeffs[i, j]).any():
                in_slit = False
            else:
                fit = edge_model(x)
                in_slit &= (fit > 0) & (fit < maxrow)

        x = x[in_slit]
        if x.size != 0:
            xranges[i] = x.min(), x.max()

    return edgecoeffs, xranges
