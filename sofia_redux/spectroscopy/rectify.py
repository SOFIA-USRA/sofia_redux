# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.spectroscopy.rectifyorder import rectifyorder

__all__ = ['rectify']


def rectify(image, ordermask, wavecal, spatcal, header=None,
            variance=None, mask=None, bitmask=None, orders=None,
            x=None, y=None, dw=None, ds=None, badfrac=0.1, ybuffer=3,
            xbuffer=None, poly_order=3):
    """
    Construct average spatial profiles over multiple orders

    A simple wrapper for `sofia_redux.spectroscopy.rectifyorder`.
    Performs minimal argument checks on arguments relating to orders.
    The rest of the checks are performed by
    `sofia_redux.spectroscopy.rectifyorder`.

    Parameters
    ----------
    image : numpy.ndarray of float (nrow, ncol)
        2-d image
    ordermask : numpy.ndarray of int (nrow, ncol)
        order number of each pixel
    wavecal : numpy.ndarray of float (nrow, ncol)
        wavelength of each pixel
    spatcal : numpy.ndarray of float (nrow, ncol)
        Spatial coordinates of each pixel
    header : fits.Header
        Header to update with spectral WCS.
    variance : numpy.ndarray of float (nrow, ncol), optional
        Variance to rectify parallel to the image.
    mask : numpy.ndarray or bool (nrow, ncol), optional
        Indicates good (True) and bad (False) pixels.
    bitmask : numpy.ndarray of int (nrow, ncol), optional
        bit-set mask
    orders : array_like of int, optional
        (norders,) array orders to process.  All are processed by default.
    x : numpy.array, optional
        (nrow, ncol) x-coordinates
    y : numpy.array, optional
        (nrow, ncol) y-coordinates
    dw : float, optional
        Delta lambda based on the span of the order in pixels and
        wavelengths.
    ds : float, optional
        The spatial sampling of the resampling slit in arcseconds,
        typically given by slth_arc / slth_pix.
    xbuffer : int, optional
        The number of pixels to ignore near the left and right of the slit.
    ybuffer : int, optional
        The number of pixels to ignore near the top and bottom of the slit.
    badfrac : float, optional
        If defines the maximum area of a pixel to be missing before
        that pixel should be considered bad.  For example, a badfrac of 0.1
        means that output flux of a pixel must be the sum of at least
        0.9 input pixels.
    poly_order : int, optional
        Polynomial order to use when converting wavecal and spatcal to
        rectified values.

    Returns
    -------
    dict
        order (int) -> dict
            image -> numpy.ndarray (ns, nw)
            wave -> numpy.ndarray (nw,)
            spatial -> numpy.ndarray (ns,)
            mask -> numpy.ndarray (ns, nw)
            bitmask -> numpy.ndarray (ns, nw)
            pixsum -> numpy.ndarray (ns, nw)
            variance -> numpy.ndarray (ns, nw)
            header -> fits.Header
    """
    ordermask = np.array(ordermask).astype(int)
    if ordermask.ndim != 2:
        log.error("Invalid ordermask dimensions")
        return
    if orders is None:
        orders = np.unique(ordermask[ordermask != 0])
    else:
        orders = np.unique(orders).astype(int)
    norders = orders.size
    if norders == 0:
        log.error("No valid orders to process")
        return

    result = {}
    for order in orders:
        result[order] = rectifyorder(
            image, ordermask, wavecal, spatcal, order,
            header=header, variance=variance,
            mask=mask, bitmask=bitmask,
            x=x, y=y, dw=dw, ds=ds, badfrac=badfrac,
            ybuffer=ybuffer, xbuffer=xbuffer,
            poly_order=poly_order)

    return result
