# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.utilities.fits import gethdul
from sofia_redux.toolkit.fitting.polynomial import poly1d
from sofia_redux.toolkit.image.adjust import rotate90

__all__ = ['readflat']


def readflat(filename):
    """
    Reads a Spextool flat field FITS image

    Reads data and metadata from an FSpextool-processed flat field
    FITS image.

    Parameters
    ----------
    filename : str
        Filename for the processed flat image

    Returns
    -------
    dict
        ``"image"``
            (numpy.ndarray) -> (nrow, ncol) image array
        ``"variance"``
            (numpy.ndarray) -> (nrow, ncol) variance array
        ``"flags"``
            (numpy.ndarray) -> (nrow, ncol) flags array
        ``"ncols"``
            (int) -> Number of columns in the image
        ``"nrows"``
            (int) -> Number of rows in the image
        ``"modename"``
            (str) -> The observing mode of the flat
        ``"slth_pix"``
            (float) -> The approximate slit height in pixels
        ``"slth_arc"``
            (float) -> The slit height in arcseconds
        ``"sltw_pix"``
            (float) -> The slit width in pixels
        ``"sltw_arc"``
            (float) -> The slit width in arcseconds
        ``"ps"``
            (float) -> The plate scale in arcseconds per pixel
        ``"rp"``
            (int) -> The resolving power
        ``"norders"``
            (int) -> Number of orders on the array
        ``"orders"``
            (list of int) -> The order numbers
        ``"edgecoeffs"``
            (numpy.ndarray) -> (norders, 2, degree+1) array of
            polynomial coefficients which define the edges of the
            orders.  edgecoeffs[0,0,:] are the coefficients of the
            bottom edge of the first order and edgecoeffs[0,1,:] are
            the coefficients of the top edge of the first order.
            Note that these coefficients are in the order required
            by numpy.poly1d (opposite order to IDL style coefficient
            ordering).
        ``"xranges"``
            (numpy.ndarray) -> (norders, 2) array of columns numbers
            between which the orders are completely on the array.
        ``"rms"``
            (numpy.ndarray) -> (norders,) An array of RMS deviations for
            each order
        ``"rotation"``
            (int) -> The rotation direction (not angle) for
            `sofia_redux.toolkit.image.adjust.rotate90`.
        ``"edgedeg"``
            (int) -> The degree of the edge coefficients
    """
    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        return

    if len(hdul) > 3:
        ishell = True
        image = hdul[1].data.copy()
        var = hdul[2].data.copy()
        flags = hdul[3].data.copy()
    else:
        ishell = False
        image, var, flags = hdul[0].data.copy(), None, None
        if image.ndim == 3:
            var = image[1].copy()
            flags = image[2].copy() if image.shape[0] >= 3 else None
            image = image[0].copy()

    hdr = hdul[0].header.copy()
    nrows, ncols = image.shape[:2]
    ps = float(hdr.get('PLTSCALE', 0))
    rp = int(hdr.get('RP', 0))
    slith_arc = float(hdr.get('SLTH_ARC', 0))
    slith_pix = float(hdr.get('SLTH_PIX', 0))
    slitw_arc = float(hdr.get('SLTW_ARC', 0))
    slitw_pix = float(hdr.get('SLTW_PIX', 0))
    rotation = int(hdr.get('ROTATION', 0))
    edgedeg = int(hdr.get('EDGEDEG', 0))
    modename = str(hdr.get('MODENAME')).strip()
    norders = int(hdr.get('NORDERS', 0))
    orders = np.array(
        [x for x in hdr.get('ORDERS', '0').split(',')
         if x != '']).astype(int)
    if rotation not in [None, 0]:
        image = rotate90(image, rotation)
        if var is not None:
            var = rotate90(var, rotation)
        if flags is not None:
            flags = rotate90(flags, rotation)

    prefix, nz = ('OR', 3) if ishell else ('ODR', 2)
    edgecoeffs = np.zeros((norders, 2, edgedeg + 1))
    rms = np.zeros((norders,))
    xranges = np.full((norders, 2), 0)
    yranges = np.full((norders, 2), 0)
    ordermask = np.full((nrows, ncols), 0)
    for orderi in range(norders):
        order = orders[orderi]
        name = prefix + str(order).zfill(nz)
        coeff_t = np.array(list(hdr['%s_T*' % name].values()))
        coeff_b = np.array(list(hdr['%s_B*' % name].values()))
        edgecoeffs[orderi, 0] = coeff_b
        edgecoeffs[orderi, 1] = coeff_t
        xranges[orderi] = np.array(
            hdr['%s_XR' % name].split(',')).astype(int)
        rms[orderi] = hdr['%sRMS' % name]

        # Calculate the order mask
        x = np.arange(xranges[orderi, 0], xranges[orderi, 1] + 1)
        botedge = poly1d(x, coeff_b)
        topedge = poly1d(x, coeff_t)
        bi, ti = botedge.astype(int), topedge.astype(int) + 1
        z = (botedge >= -0.5) & (topedge <= (nrows - 0.5))
        for j in range(int(np.ptp(xranges[orderi])) + 1):
            if z[j]:
                ordermask[bi[j]: ti[j], x[j]] = order
        yranges[orderi] = np.array([min(bi), max(ti)])

    if slith_pix > 0:
        ds = slith_arc / slith_pix
    else:
        log.warning('Slit height in pixels is 0.')
        ds = 1.0

    result = {'image': image, 'variance': var, 'flags': flags,
              'nrows': nrows, 'ncols': ncols,
              'omask': ordermask, 'edgedeg': edgedeg,
              'norders': norders, 'orders': orders,
              'edgecoeffs': edgecoeffs,
              'xranges': xranges, 'yranges': yranges,
              'ps': ps, 'rp': rp, 'rotation': rotation,
              'slith_arc': slith_arc, 'slith_pix': slith_pix,
              'slitw_arc': slitw_arc, 'slitw_pix': slitw_pix,
              'ds': ds,
              'modename': modename, 'rms': rms}

    return result
