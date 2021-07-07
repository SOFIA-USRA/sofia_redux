# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings
from astropy import log
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap

from sofia_redux.instruments.forcast.clean import clean
from sofia_redux.instruments.forcast.jbclean import jbclean

addhist = add_history_wrap('Darksum')

__all__ = ['darksum']


def darksum(dark, header, darkvar=None, badmap=None, jailbar=False,
            extra=None):
    """
    Creates the master dark frame

    Cleans, averages, and jailbar-corrects individual dark frames to make
    a single master dark frame.  Calls `clean` and `jbclean`.  The JAILBAR
    parameter should only be set if the JBCLEAN method is 'median';
    otherwise, `jbclean` is called from within `clean`.  The return value
    is a dictionary that contains the final master dark, as well as its
    variance array, and the intermediate cleaned dark and its variance
    array.

    Parameters
    ----------
    dark : numpy.ndarray
        Array containing the dark frames (nframe, nrow, ncol)
    header : astropy.io.fits.header.Header
        The FITS header of the dark file; will be updated with HISTORY
        messages
    darkvar : numpy.ndarray, optional
        Variance of the dark frames (nframe, nrow, ncol) to update in
        parallel with the data array
    badmap : numpy.ndarray, optional
        (nframe, nrow, ncol).  If not set, bad pixels will not be cleaned
    jailbar : bool, optional
        If True, `jbclean` will be called on the averaged dark frames
    extra : dict, optional
        If provided, will be updated with 'cleaned' and 'cleanedvar'
        containing (nrow, ncol) arrays or the darks and variance after
        correction of the bad pixels.
    Returns
    -------
    2-tuple
        numpy.ndarray - Averaged dark frame (nrow, ncol)
        numpy.ndarray - Propagated variance (nrow, ncol)
    """

    # define a null array consistent with the input dark
    if not isinstance(dark, np.ndarray):
        log.error("dark is not a %s" % np.ndarray)
        return

    if np.isnan(dark).all() or np.nanmax(np.abs(dark)) == 0:
        log.error("No Dark Frames")
        return

    # First, clean the bad pixels if required or initialize cleaned and
    # cleanedvar with null array
    cleaned = dark.copy()
    dovar = isinstance(darkvar, np.ndarray) and darkvar.shape == dark.shape
    if isinstance(darkvar, np.ndarray) and dovar:
        log.warning("invalid variance")
    cleanedvar = darkvar.copy() if dovar else None

    if isinstance(badmap, np.ndarray):
        log.info("cleaning bad pixels from darks")
        cleaning = clean(dark, badmap, header, variance=darkvar)
        if cleaning is None:
            addhist(header, 'cleaning failed')
            log.error('cleaning failed')
        else:
            cleaned, cleanedvar = cleaning
            log.info("done cleaning bad pixels")
            if isinstance(extra, dict):
                extra['cleaned'] = cleaned
                extra['cleanedvar'] = cleanedvar

    # Calculate the average of darks.  If it is a 2D aray, the sum is
    # just the dark.  If not, we average the planes.
    log.info("summing darks")
    if len(dark.shape) > 2:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dsum = np.nanmean(cleaned, axis=0)
            if dovar:
                dsumvar = np.nansum(cleanedvar, axis=0)
                totvar = np.sum(~np.isnan(cleanedvar), axis=0).astype(int)
                totvar **= 2
                zi = dsumvar == 0
                dsumvar[~zi] /= totvar[~zi]
                dsumvar[zi] = np.nan
            else:
                dsumvar = None
    else:
        dsum = cleaned.copy()
        dsumvar = cleanedvar.copy() if dovar else None

    # perform jailbar correction if requested
    result = dsum, dsumvar
    if jailbar:
        log.info("cleaning jailbar from darks")
        result = jbclean(result[0], header, variance=result[1])
        log.info("done cleaning jailbar")

    return result
