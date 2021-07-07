# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap
from sofia_redux.toolkit.image import adjust

from sofia_redux.instruments.forcast.getpar import getpar

addhist = add_history_wrap('Shift')

__all__ = ['shift']


def symmetric_ceil(val):
    def f(x):
        return int(np.ceil(x)) if x >= 0 else int(np.floor(x))
    if not hasattr(val, '__len__'):
        return f(val)
    else:
        return [f(v) for v in val]


def shift(data, offset, header=None, variance=None, order=None,
          crpix=None, resize=False, no_shift=False, missing=np.nan,
          **kwargs):
    """
    Shift an image by the specified amount

    Uses interpolation to do sub-pixel shifts if desired.  Missing
    data should be represented with numpy.nans.

    Parameters
    ----------
    data : numpy.ndarray
        The image array to shift (nrow, ncol)
    offset : array_like
        The (x, y) offset by which to shift the image.
    header : astropy.io.fits.header.Header
        Header update with CRPIX
    variance :  numpy.ndarray, optional
        Variance array (nrow, ncol) to update in parallel with the data
        output data array
    order : int, optional
        Interpolation order.
            0 - nearest-neighbor
            1 - bilinear
            >=2 - spline of the same order
    crpix : array_like, optional
        If provided, will be updated to match image shift_image
        [crpix1, crpix2]
    resize : bool, optional
        Increase the size of the data array to accommodate offset
    no_shift : bool, optional
        If True, do not shift.  Just return the original resized
        arrays prior to any shift.
    missing
        missing data fill value

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The shifted image (nrow, ncol)
        The shifted variance (nrow, ncol) or None if not supplied

    Notes
    ------
    Please see sofia_redux.toolkit.image.adjust.shift for full descriptions of
    available kwargs.  The most useful parameters are listed below:

        missing
            value with which to replace missing values (NaNs)
        mode : str
            edge handling mode.  Default is 'constant'

    """
    if not isinstance(header, fits.header.Header):
        header = fits.header.Header()
    var = variance.copy() if isinstance(variance, np.ndarray) else None
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        addhist(header, 'Shift failed (invalid data)')
        log.error("invalid data")
        return
    shifted = data.copy()

    if order is None:
        order = getpar(header, 'SHIFTORD', dtype=int, default=1,
                       comment='Interpolate order for coadd/merge')

    if crpix is not None:
        if not hasattr(crpix, '__len__') or len(crpix) != 2:
            log.error("invalid crpix - will not update")
            crpix = np.flip((np.array(shifted.shape) + 1) / 2)
    else:
        crpix = np.flip((np.array(shifted.shape) + 1) / 2)

    dovar = isinstance(var, np.ndarray) and var.shape == shifted.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Variance not propagated (invalid variance)')
        log.warning("Variance not propagated (invalid variance)")

    if not hasattr(offset, '__len__') or len(offset) != 2:
        log.error("invalid offset")
        return

    minxy = np.array([0, 0])  # additional offsets for resizing
    if resize:
        ixy = symmetric_ceil(offset)
        resize_shape = (shifted.shape[0] + abs(ixy[1]),
                        shifted.shape[1] + abs(ixy[0]))
        if any(ixy):
            addhist(header, "Increasing (X,Y) array size to (%s,%s)" %
                    (resize_shape[1], resize_shape[0]))
            header['NAXIS1'] = resize_shape[1]
            header['NAXIS2'] = resize_shape[0]
            newarray = np.full(resize_shape, missing, dtype=shifted.dtype)
            # lower-left (x, y) corner
            minxy = np.array([0 if x > 0 else -x for x in ixy])
            xl, xu = minxy[0], minxy[0] + shifted.shape[1]
            yl, yu = minxy[1], minxy[1] + shifted.shape[0]
            newarray[yl: yu, xl: xu] = shifted
            shifted = newarray
            if dovar:
                newarray = np.full(resize_shape, missing, dtype=shifted.dtype)
                newarray[yl: yu, xl: xu] = var
                var = newarray

    # Offset due to resizing
    if 'CRPIX1' in header:
        header['CRPIX1'] += minxy[0]
    if 'CRPIX2' in header:
        header['CRPIX2'] += minxy[1]
    if 'SRCPOSX' in header:
        header['SRCPOSX'] += minxy[0]
    if 'SRCPOSY' in header:
        header['SRCPOSY'] += minxy[1]
    crpix[0] += minxy[0]
    crpix[1] += minxy[1]
    if no_shift:
        return shifted, var

    if any(offset):
        shifted = adjust.shift(shifted, np.flip(offset), order=order,
                               missing=missing, **kwargs)
        if shifted is None:
            log.error("shift failed (data)")
            return
        if dovar:
            # always do nearest neighbor for error -- otherwise,
            # we get weird edge effects when combining images
            # and weighting by errors
            var = adjust.shift(var, np.flip(offset), order=0,
                               missing=missing, **kwargs)

    # offset due to shifting
    if 'CRPIX1' in header:
        header['CRPIX1'] += offset[0]
    if 'CRPIX2' in header:
        header['CRPIX2'] += offset[1]
    if 'SRCPOSX' in header:
        header['SRCPOSX'] += offset[0]
    if 'SRCPOSY' in header:
        header['SRCPOSY'] += offset[1]
    crpix[0] += offset[0]
    crpix[1] += offset[1]

    return shifted, var
