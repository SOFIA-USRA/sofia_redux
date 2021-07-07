# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap

from sofia_redux.instruments.forcast.getpar import getpar

addhist = add_history_wrap('Flat')

__all__ = ['flat']


def flat(data, flatsum, darksum=None, header=None, variance=None,
         flatvar=None, darkvar=None, missing=np.nan):
    """
    Flat field corrects data frames

    Dark and flatfield corrects data frames.  A master flat field image
    has to be created beforehand by cleaning and summing individual flat
    field images and dividing the sum by its median.  A master dark
    should also be created by averaging individual dark frames.  The
    masterdark is subtracted from each member of the input data set,
    then the result is divided by the masterflat.  If variance is
    passed, flatvar and darkvar should be provided as well.  This
    function is currently disabled for imaging data, pending further
    information from the instrument team: if called, and the ICONFIG
    keyword in dripconf.txt is not 'SPECTROSCOPY', it will simply add
    a message to the header array saying that the correction was not
    applied, then return the input data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (nimage, nrow, ncol)
    flatsum : numpy.ndarray
        Normalized master flat field array (nrow, ncol)
    darksum : numpy.ndarray, optional
        Master dark array (nrow, ncol).  If provided, will be subtracted
        from the data prior to flat correction.  Not typically used, as
        frames are usually subtracted, which removes the dark signal
        automatically.
    header : astropy.io.fits.header.Header, optional
        Header to update with a HISTORY message
    variance : numpy.ndarray, optional
        Variance array (nimage, nrow, ncol) to update in parallel with the
        data array.
    flatvar : numpy.ndarray, optional
        Flat variance array (nrow, ncol) to propagate.  Must be provided
        if variance is set.
    darkvar : numpy.ndarray, optional
        Dark variance array (nrow, ncol) to propagate.  Must be provided
        if variance is set and darksum is provided.
    missing : float, optional
        value to replace NaNs with in final product

    Returns
    -------
    numpy.array, numpy.array
        The flat-corrected data array (nframe, nrow, ncol)
        The propagated variance array (nframe, nrow, ncol)
    """
    if header is None:
        header = fits.header.Header()
        addhist(header, 'Created header')

    # Imaging flats are not currently used
    specmode = getpar(header, 'ICONFIG', dtype=str).upper().strip()
    if specmode != 'SPECTROSCOPY':
        addhist(header, 'Flatfield was not applied')
        log.warning("IMAGE flat correction is not yet available")
        return
    else:
        addhist(header, 'Flatfield applied')

    if not isinstance(data, np.ndarray) or len(data.shape) not in [2, 3]:
        addhist(header, 'Not applied (invalid data)')
        log.error("must provide valid data array")
        return
    imshape = data.shape[-2:]

    doflat = isinstance(flatsum, np.ndarray) and flatsum.shape == imshape
    if not doflat:
        addhist(header, 'Not applied (invalid master flat)')
        log.error("must provide valid master flat")
        return

    dodark = isinstance(darksum, np.ndarray) and darksum.shape == imshape
    if darksum is not None and not dodark:
        # An error if dark was supplied but incorrect
        addhist(header, 'Not applied (invalid master dark)')
        log.error("master dark does not match image frames")
        return

    dovar, var = False, None
    if variance is not None:
        varok = isinstance(variance, np.ndarray) and \
            variance.shape == data.shape
        if not varok:
            addhist(header, 'Not propagating variance (invalid variance)')
            log.error("variance must match data array")
        fvarok = isinstance(flatvar, np.ndarray) and \
            flatvar.shape == imshape
        if not fvarok:
            addhist(header, 'Not propagating variance (invalid flat variance')
            log.error("flat variance does not match data frames")
        dvarok = isinstance(darkvar, np.ndarray) and \
            darkvar.shape == imshape
        if dodark and not dvarok:
            addhist(header, 'Not propagating variance (invalid dark variance')
            log.error("dark variance does not match data frames")
        dovar = varok and fvarok
        dovar = dvarok if dodark else dovar

    flatted, flatsum2 = data.copy(), None
    if dovar:
        var = variance.copy()
        flatsum2 = flatsum ** 2

    zi = (np.isnan(flatsum)) | (flatsum == 0)
    if len(data.shape) == 2:
        flatted = np.array([flatted])
        var = np.array([var])
    else:
        if var is None:
            var = [var] * data.shape[0]

    for frame, vframe in zip(flatted, var):
        if dodark:
            frame -= darksum
        frame[~zi] /= flatsum[~zi]
        frame[zi] = missing
        if dovar:
            if dodark:
                vframe += darkvar
            vframe += (frame ** 2) * flatvar
            vframe[~zi] /= flatsum2[~zi]
            vframe[zi] = missing

    if len(data.shape) == 2:
        flatted, var = flatted[0], var[0]
    if not dovar:
        var = None

    return flatted, var
