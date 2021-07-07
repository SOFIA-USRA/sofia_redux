# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap, hdinsert, kref

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.shift import shift

addhist = add_history_wrap('Merge')

__all__ = ['merge_shift']


def merge_shift(data, chopnod, header=None, variance=None, nmc=False,
                maxshift=999999999., normmap=None, resize=True):
    """
    Merge an image by shifting the input data by the input values

    Add each frame of the data to a 2-d summation frame in a manner
    appropriate to the current reduction scheme.  Finally, average
    by the number of frames.

    Parameters
    ----------
    data : numpy.ndarray
        Data to be merged i.e. frame with target images (nrow, ncol)
    chopnod : array-like
        Chop/Nod shifts [chopx, chopy, nodx, nody]
    header : astropy.io.fits.header.Header, optional
        FITS header to update
    variance : numpy.ndarray, optional
        Propagate provided variance (nrow, ncol)
    nmc : bool, optional
        Set to True if NMC image
    maxshift : float, optional
        Will not merge if nod or chop distance is greater than
        maxshift.
    normmap : numpy.ndarray, optional
        Array to hold the normalization map
    resize : bool, optional
        If True, resize the output result to accomodate shifting

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        - The merged image (nrow, ncol) i.e. frame with images of
          the object at chop and nod positions merged.
        - The propagated variance image (nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        header = fits.header.Header()
        addhist(header, 'Created header')

    var = variance.copy() if isinstance(variance, np.ndarray) else None
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        addhist(header, "chop positions not applied (invalid data)")
        log.error("invalid data")
        return
    elif np.isnan(data).all():
        addhist(header, "chop positions not applied (invalid data)")
        log.error("data are all NaN")
        return

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning("invalid variance")

    if not hasattr(chopnod, '__len__') or len(chopnod) != 4 or \
            np.isnan(chopnod).any():
        addhist(header, "chop positions not applied (invalid chopnod)")
        log.error("invalid chop nod")
        return

    if isinstance(normmap, np.ndarray):
        # normmap is not critical, so we can resize
        if normmap.shape != data.shape:
            normmap.resize(data.shape, refcheck=False)
        normmap.fill(0)
    else:
        normmap = np.zeros_like(data)

    chopnod = np.array(chopnod)
    chopdist = np.sqrt((chopnod[:2] ** 2).sum())
    merged = data.copy()
    # Replace NaNs wih zeroes for adding
    nanidx = np.isnan(merged)
    merged[nanidx] = 0
    normmap[merged != 0] = 1
    if dovar:
        var[nanidx] = 0

    # order=0 results in nearest-neighbor interpolation
    order = getpar(header, 'SHIFTORD', dtype=int, default=0,
                   comment='interpolate order for coadd/merge')
    addhist(header, 'Shift interpolation order is %i' % order)

    # Shift by chop
    if chopdist > maxshift:
        merged = data.copy()
        if nmc:
            # Still need to divide the central source by 2 for NMC
            merged /= 2
            normmap[merged != 0] += 1
            if dovar:
                var /= 4
        addhist(header, 'chop positions was not applied '
                        '(chop greater than %f)' % maxshift)
        return merged, var

    addhist(header, 'X, Y chop shifts are %f,%f' %
            (chopnod[0], chopnod[1]))
    hdinsert(header, 'MRGDX0', chopnod[0], refkey=kref,
             comment='X shift during merge process')
    hdinsert(header, 'MRGDY0', chopnod[1], refkey=kref,
             comment='Y shift during merge process')

    merged0 = merged.copy()
    normmap0 = normmap.copy()
    var0 = var.copy() if dovar else None
    if resize:
        log.info("Resizing for chop shifts")
        addhist(header, 'Resizing for chop shifts')
        merged, _ = shift(merged0, chopnod[:2], order=order, header=header,
                          missing=0, resize=True, no_shift=True)
        normmap.resize(merged.shape, refcheck=False)
        normmap[:, :], var = shift(normmap0, chopnod[:2], order=0,
                                   variance=var0, missing=0, resize=True,
                                   no_shift=True)

    chop_data, _ = shift(merged0, chopnod[:2], order=order, missing=0,
                         resize=resize)
    chop_mask, chop_var = shift(normmap0, chopnod[:2], order=0,
                                variance=var0, resize=resize, missing=0)

    if not nmc:
        # the nodding shift is cumulative to the chop shift
        merged -= chop_data
        normmap += chop_mask
        if dovar:
            var += chop_var

    # Shift by nod
    noddist = np.sqrt((chopnod[2:] ** 2).sum())
    nod = nmc or (noddist <= maxshift and noddist != 0)
    nod_data = np.zeros_like(merged)
    nod_mask = np.zeros_like(normmap)
    nod_var = np.zeros_like(var) if dovar else None
    if not nod:
        addhist(header, 'nod positions was not applied '
                        '(nod greater than %f)' % maxshift)
    else:
        addhist(header, 'X, Y nod shifts are %f,%f' %
                (chopnod[2], chopnod[3]))
        hdinsert(header, 'MRGDX1', chopnod[2], refkey=kref,
                 comment='X shift during merge process')
        hdinsert(header, 'MRGDY1', chopnod[3], refkey=kref,
                 comment='Y shift during merge process')
        merged0 = merged.copy()
        normmap0 = normmap.copy()
        var0 = var.copy() if dovar else None
        if resize:
            log.info("Resizing for nod shifts")
            addhist(header, 'Resizing for nod shifts')
            merged, _ = shift(merged0, chopnod[2:], order=order, header=header,
                              missing=0, resize=True, no_shift=True)
            normmap.resize(merged.shape, refcheck=False)
            normmap[:, :], var = shift(normmap0, chopnod[2:], order=0,
                                       variance=var0, missing=0, resize=True,
                                       no_shift=True)
            chop_data, _ = shift(chop_data, chopnod[2:], order=order,
                                 missing=0, resize=True, no_shift=True)
            chop_mask, chop_var = shift(chop_mask, chopnod[2:], order=0,
                                        variance=chop_var, missing=0,
                                        resize=True, no_shift=True)

        nod_data, _ = shift(merged0, chopnod[2:], order=order, missing=0,
                            resize=resize)
        nod_mask, nod_var = shift(normmap0, chopnod[2:], order=0,
                                  variance=var0, resize=resize, missing=0)

    if not nmc:
        # the shifting has been cumulative
        # i.e. result = (data - chop(data)) - nod(data - chop(data))
        merged -= nod_data
        normmap += nod_mask
        if dovar:
            var += nod_var
    else:
        # the shifting was not cumulative i.e.
        # i.e. result = data - chop(data) - nod(data)
        normmap += chop_mask + nod_mask
        merged -= chop_data + nod_data
        if dovar:
            var += chop_var + nod_var

    # Normalize by merge mask
    idx = normmap != 0
    if nmc:
        # Add one extra source for NMC
        normmap[idx] += 1

    merged[idx] /= normmap[idx]
    if dovar:
        var[idx] /= normmap[idx] ** 2

    # Put NaNs back in
    zi = normmap == 0
    merged[zi] = np.nan
    if dovar:
        var[var == 0] = np.nan

    return merged, var
