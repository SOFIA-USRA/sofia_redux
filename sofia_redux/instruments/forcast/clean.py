# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io.fits.header import Header
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap
from sofia_redux.toolkit.image.fill import maskinterp

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.jbclean import jbclean

addhist = add_history_wrap('Clean')

__all__ = ['clean']


def clean(data, badmap=None, header=None, variance=None,
          propagate_nan=False, **kwargs):
    """
    Replaces bad pixels in an image with approximate values

    Interpolates over bad values.  If the clean method used for jailbar
    pattern removal is FFT, it is applied here.  If it is median, it is
    applied to the science in `stack` or to the calibrations in `getcal`.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (nimage, nrow, ncol)
    badmap : numpy.ndarray, optional
        Bad pixel map (nrow, ncol) of bools.
        False = good pixel, True = bad pixel
    header : astropy.io.fits.header.Header
        Input header, will be updated with HISTORY messages
    variance : numpy.ndarray, optional
        Variance array (nimage, ncol, nrow) to update in parallel
        with the data array
    propagate_nan : bool, optional
        If set, bad pixels will be set to NaN instead of interpolated
        over.
    kwargs
        Optional parameters to pass into mask interp.  The most relevant
        default settings are maxap=6, order=3.  See interpolate.maskinterp
        for further details

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Cleaned data array (nimage, ncol, nrow)
        Propagated variance array (nimage, ncol, nrow)
    """
    if not isinstance(header, Header):
        header = Header()
        addhist(header, 'Created header')

    var = variance.copy() if isinstance(variance, np.ndarray) else None

    if not isinstance(data, np.ndarray) or len(data.shape) not in [2, 3]:
        addhist(header, 'Did not clean bad pixels (Invalid data)')
        log.error("data is not a valid array: %s" % type(data))
        return
    cleaned = data.copy()

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning("Variance must match data: %s" % type(variance))

    if not isinstance(badmap, np.ndarray) or len(badmap.shape) != 2 or \
            badmap.shape != data.shape[-2:]:
        addhist(header, 'Did not clean bad pixels (Invalid mask)')
        log.warning("bad pixel map is not a valid array - will not clean")
        badmap = None
    else:
        badmap = badmap.astype(bool)

    nimage = 1 if len(data.shape) == 2 else data.shape[0]
    if nimage == 1:
        cleaned = np.array([cleaned])
        var = np.array([var])
    elif not dovar:
        var = np.array([None] * nimage)

    # Set NaNs for bad pixels
    if badmap is not None:
        for frame, v in zip(cleaned, var):
            frame[badmap] = np.nan
            if dovar:
                v[badmap] = np.nan

    # better results if JBCLEAN is called before maskinterp
    jbmethod = getpar(header, 'JBCLEAN', dtype=str, default=None,
                      comment='Jail bar cleaning algorithm')
    jbmethod = jbmethod.strip().upper()
    addhist(header, 'Jailbar cleaning method is %s' % jbmethod)
    if jbmethod == 'FFT':
        for idx in range(nimage):
            cleaning = jbclean(cleaned[idx], header=header, variance=var[idx])
            if cleaning is not None:
                cleaned[idx], var[idx] = cleaning
                log.info('Jailbars cleaned from frame %i' % (idx + 1))
            else:
                addhist(header, 'failed on frame %i' % (idx + 1))
                log.error('Jailbar cleaning failed on frame %i' % (idx + 1))

    if badmap is not None:
        if not propagate_nan:
            addhist(header, 'Interpolate using maskinterp')
        else:
            addhist(header, 'Masking bad values with NaN')
        for idx in range(nimage):
            if not propagate_nan:
                mask = ~badmap.astype(bool) & ~np.isnan(cleaned[idx])
                cleaned[idx] = maskinterp(cleaned[idx],
                                          mask=mask, **kwargs)
                if dovar:
                    mask = ~badmap.astype(bool) & ~np.isnan(var[idx])
                    var[idx] = maskinterp(var[idx],
                                          mask=mask, **kwargs)
            else:
                cleaned[idx][badmap.astype(bool)] = np.nan
                if dovar:
                    var[idx][badmap.astype(bool)] = np.nan

    # The original code states that the top line is corrupted by clean.
    # It seems that the data is already bad - clean is not responsible.
    # We replace the top line by the second from top line
    ny = data.shape[1]
    if not propagate_nan:
        cleaned[:, ny - 1, :] = cleaned[:, ny - 2, :].copy()
        if dovar:
            var[:, ny - 1, :] = var[:, ny - 2, :].copy()
    else:
        cleaned[:, ny - 1, :] = np.nan
        if dovar:
            var[:, ny - 1, :] = np.nan

    if nimage == 1:
        cleaned = cleaned[0]
        var = var[0]
    elif None in var:
        var = None

    return cleaned, var
