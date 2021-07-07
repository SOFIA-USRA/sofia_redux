# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.fitting.polynomial import poly1d

__all__ = ['adjustguesspos']


def adjustguesspos(edgecoeffs, xranges, flat, ordermask, orders=None,
                   cor_order=None, ybuffer=0, default=False):
    """
    Adjust the guess positions via cross-correlation

    Performs a vertical cross-correlation between the order mask and
    the raw flat field of a single order to identify the shift between
    the two.  This shift is subtracted from the guess positions.  The
    xranges are then checked to ensure the order range does not fall
    off the array.

    Notes
    -----
    If the range falls off the array, this will be marked by
    xrange = [-1, -1] for the affected order.

    Parameters
    ----------
    edgecoeffs : array_like of float (norders, 2, order+1)
        Polynomial coefficients which define the edges of the orders.
        edgecoeffs[0, 0, :] are the coefficients of the bottom edge of
        the first order and edgecoeffs[0, 1, :] are the coefficients
        of the top edge of the first order.
    xranges : array_like of float (norders, 2)
        Column numbers between which the orders are completely on the array.
    flat : array_like of float (nrow, ncol)
        Raw flat field
    ordermask : array_like of int (nrow, ncol)
        Array where each pixel is set to its order number.  Interorder
        pixels are set to zero.
    orders : array_like of int (norders,), optional
        Order numbers to process.  Set to all by default.
    cor_order : int, optional
        The order with which o do the cross-correlation.  Will default
        to the first order in `orders`.
    ybuffer : int, optional
        The number of pixels to buffer from the top and bottom of the
        array.
    default : bool, optional
        If True, return the default guess positions and xranges.

    Returns
    -------
    guess, xranges : numpy.ndarray, numpy.ndarray
        The adjusted guess positions and xranges.  Both of type int
        and size (norders, 2).  guess[:, 0] = y-guess and
        guess[:, 1] = x-guess.

    """
    edgecoeffs = np.array(edgecoeffs).astype(float)
    if edgecoeffs.ndim != 3 or edgecoeffs.shape[1] != 2:
        log.error("Invalid edgecoeffs shape")
        return
    xranges = np.array(xranges).astype(int)
    if xranges.ndim != 2 or xranges.shape[1] != 2:
        log.error("Invalid xranges shape")
        return
    if xranges.shape[0] != edgecoeffs.shape[0]:
        log.error("Edgecoeffs and xranges order size shape mismatch")
        return
    flat = np.array(flat).astype(float)
    if flat.ndim != 2:
        log.error("Invalid flat shape")
        return
    shape = flat.shape
    ordermask = np.array(ordermask).astype(int)
    if ordermask.shape != shape:
        log.error("Flat and ordermask shape mismatch")
        return
    if orders is None:
        orders = np.unique(ordermask[ordermask != 0]).astype(int)
    else:
        orders = np.unique(orders).astype(int)
    norders = orders.size
    if cor_order is None:
        cor_order = orders[0]

    # Compute the guess positions
    guess = np.zeros((norders, 2), dtype=int)

    for orderi in range(norders):
        xcenter = np.mean(xranges[orderi])
        botedge = poly1d(xcenter, edgecoeffs[orderi, 0])
        topedge = poly1d(xcenter, edgecoeffs[orderi, 1])
        ycenter = np.mean([botedge, topedge])
        guess[orderi] = np.round([ycenter, xcenter])

    if default:
        return guess, xranges

    # Begin cross-correlation
    omask = np.array(ordermask == cor_order)
    orderi = np.argmax(orders == cor_order)

    x = np.arange(xranges[orderi, 0], xranges[orderi, 1] + 1)
    botedge = poly1d(x, edgecoeffs[orderi, 0])
    topedge = poly1d(x, edgecoeffs[orderi, 1])
    slith_pix = int(np.ceil(np.max(topedge - botedge)))

    # Determine the top and bottom row of the subimage to clip out
    y0 = np.clip(int(np.round(np.min(botedge) - slith_pix)), 0, shape[0])
    y1 = np.clip(int(np.round(np.max(topedge) + slith_pix)), 0, shape[0]) + 1
    subflat, subomask = flat[y0:y1], omask[y0:y1]

    # shifts are in the y-direction only
    subshape = subflat.shape
    nshifts = slith_pix * 2 + 1
    shifts = np.arange(nshifts) - slith_pix
    corr = np.zeros(nshifts)

    for i, s in enumerate(shifts):
        hbot = np.clip(-s, 0, subshape[0])
        htop = np.clip(subshape[0] - s, 0, subshape[0])
        mbot = np.clip(s, 0, subshape[0])
        mtop = np.clip(subshape[0] + s, 0, subshape[0])
        corr[i] = np.sum(subflat[hbot:htop] * subomask[mbot:mtop])

    offset = shifts[np.argmax(corr)]
    guess[:, 0] -= offset

    # now check xranges
    dy = ybuffer - 1
    for orderi in range(norders):
        xrange = xranges[orderi]
        x = np.arange(xrange[1] - xrange[0] + 1) + xrange[0]
        botedge = poly1d(x, edgecoeffs[orderi, 0]) - offset
        topedge = poly1d(x, edgecoeffs[orderi, 1]) - offset
        idx = (botedge > dy) & (topedge < (shape[0] - dy))
        if not idx.any():
            log.warning("Order %i shifted past y-buffer" % orders[orderi])
            xrange[:] = -1, -1
        else:
            xrange[:] = np.min(x[idx]), np.max(x[idx])

    return guess, xranges
