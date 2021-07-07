# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re
from warnings import simplefilter, catch_warnings

from astropy import log
from astropy.io import fits
import numpy as np
import pandas

from sofia_redux.toolkit.utilities.fits import add_history_wrap
from sofia_redux.toolkit.image.fill import maskinterp

import sofia_redux.instruments.forcast as drip
from sofia_redux.instruments.forcast.background import background
from sofia_redux.instruments.forcast.clean import clean
from sofia_redux.instruments.forcast.droop import droop
from sofia_redux.instruments.forcast.imgnonlin import imgnonlin
from sofia_redux.instruments.forcast.jbclean import jbclean
from sofia_redux.instruments.forcast.getdetchan import getdetchan
from sofia_redux.instruments.forcast.read_section import read_section

addhist = add_history_wrap('Flatsum')

__all__ = ['clean_flat', 'get_flatbias', 'create_master_flat', 'flatsum']


def clean_flat(flat, header=None, variance=None, badmap=None, jailbar=False):
    """
    Clean flat based on bad pixel mask, optionally remove jailbars

    This is a very strange procedure.  If JBCLEAN='FFT' then `clean`
    will remove the jailbars automatically.  If jailbar=True, then
    the jailbar removal will occur for a second time.

    Note that jailbar cleaning was performed after interpolating over
    NaNs in the previous version.  The results are MUCH better in
    testing if performed the other way around.  Therefore, that's
    what I've done here.

    Parameters
    ----------
    flat : numpy.ndarray
        flat to be cleaned (nframe, nrow, ncol) or (nrow, ncol)
    header : astropy.io.fits.header.Header, optional
        FITS header
    variance : numpy.ndarray, optional
        variance array (nrow, ncol) to update in parallel
    badmap : numpy.ndarray, optional
        defines bad pixel mask (nrow, ncol).
        True = bad pixel, False = good pixel
    jailbar : bool, optional
        remove jailbars using sofia_redux.instruments.forcast.jbclean

    Returns
    -------
    numpy.ndarray
        The cleaned flat data array (nrow, ncol)
    """
    if not isinstance(flat, np.ndarray):
        return
    data = flat.copy()
    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    var = variance.copy() if dovar else None
    ndim = len(data.shape)
    if ndim == 2:
        data = np.array([data])
        var = np.array([var])
    elif not dovar:
        var = np.array([None] * data.shape[0])

    if badmap is not None:
        for d, v in zip(data, var):
            d[badmap] = np.nan
            if dovar:
                v[badmap] = np.nan

    if jailbar:
        log.info("cleaning jailbar from flats")
        for idx, (d, v) in enumerate(zip(data, var)):
            result = jbclean(d, header=header, variance=v)
            if result is None:
                msg = "Jailbar removal failed at clean_flat"
                addhist(header, msg)
                log.error(msg)
                return
            data[idx], var[idx] = result[0], result[1]

    if ndim == 2:
        data, var = data[0], var[0]
    elif not dovar:
        var = var[0]

    result = data, var
    if isinstance(badmap, np.ndarray):
        log.info("cleaning bad pixels from flats")
        result = clean(result[0], badmap, header, variance=result[1])

    return result


def get_flatbias(header, pathcal=None):
    """
    Retrieve bias levels from biaslevels.txt file based

    Header contents are used to select detector channel and
    capacitance.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        FITS header
    pathcal : str, optional
        Path to the data directory containing biaslevels.txt
        file.

    Returns
    -------
    int
        bias level retieved from bias file
    """
    # get date of observation
    date_arr = re.split('[-T]', header.get('DATE-OBS', ''))
    fdate = 99999999
    if len(date_arr) >= 3:
        datestr = ''.join(date_arr[:3])
        try:
            fdate = int(datestr)
        except (TypeError, ValueError):
            pass

    # get mode from flat
    detchan = getdetchan(header)
    ilowcap = header.get('ILOWCAP')
    detchan = 'lwc' if detchan == 'LW' else 'swc'
    ilowcap = 'hi' if not ilowcap else 'lo'

    # read bias levels from a table
    if not isinstance(pathcal, str):
        pathcal = os.path.join(os.path.dirname(drip.__file__), 'data')
    blfile = os.path.join(pathcal, 'biaslevels.txt')
    if os.path.isfile(blfile):
        bias_levels = pandas.read_csv(
            blfile, delim_whitespace=True, comment='#',
            names=['date', 'swclo', 'swchi', 'lwclo', 'lwchi'])
        rows = bias_levels.loc[bias_levels['date'] >= fdate]
        if len(rows) > 0:
            levels = rows.iloc[0]
            return levels[detchan + ilowcap]
    return 0


def create_master_flat(data, header, variance=None, dark=None, darkvar=None,
                       ordermask=None, pathcal=None, normflat=False, **kwargs):
    """
    Create a master flat

    Removes the dark or bias from the flat data, update the variance.

    Parameters
    ----------
    data : numpy.ndarray
        input flat data (nframe, nrow, ncol) or (nrow, ncol)
    header : astropy.io.fits.header.Header
    variance : numpy.ndarray, optional
        variance array to update in parallel. (nframe, nrow, ncol) or
        (nrow, ncol).  Will be collapsed to (nrow, ncol) if 3-dimensional
        on input.
    dark : numpy.ndarray, optional
        dark array to subtract (nrow, ncol)
    darkvar : numpy.ndarray, optional
        dark variance to propagate (nrow, ncol)
    ordermask : numpy.ndarray, optional
        Mask indicating edges of orders in spetroscopic flats.
        (>=1, True)=good, (0, False)=bad pixel.  Areas inside orders
        will be used to normalize the flat; areas outside the orders
        will be set to 1.0 in the masterflat.
    pathcal : str, optional
        Path to the data directory.  Used to find the
        default bias leves if darks are not passed.
    normflat : bool, optional
        If True, the output frame will be normalizes by the median
        of all flat data where ordermask=1.  If unset, the output
        frame will not be normalized.
    kwargs
        Optional parameters to pass into
        sofia_redux.instruments.forcast.interpolate.maskinterp.
        Note that func=numpy.nanmedian is hard-coded into this
        implementation.

    Returns
    -------
    2-tuple
        numpy.ndarray : master flat (nrow, ncol)
        numpy.ndarray : master variance (nrow, ncol)

    Notes
    -----
    When subtracting the dark note that this is only strictly correct
    if the dark exposure time matches the flat exposure time.  Otherwise,
    the dark should have the bias removed, then be scaled to the exposure
    time of the flat.  However, for the typically short exposure times
    used for flats, the dark current is negligible, and the dark frame
    can be treated as a bias frame.
    """
    darksub = False
    if isinstance(dark, np.ndarray):
        if dark.shape != data.shape[-2:]:
            log.warning("dark shape does not match flat shape")
            log.warning("not subtracting dark current")
        else:
            darksub = True
            if isinstance(darkvar, np.ndarray):
                if darkvar.shape != data.shape[-2:]:
                    log.warning("darkvar shape does not match flat shape")
                    log.warning("not propagating dark variance")
                    darkvar = 0
            else:
                darkvar = 0

    # Build master flatfield image
    dovar = isinstance(variance, np.ndarray)
    diff = data.copy()
    diffvar = variance.copy() if dovar else None

    if len(data.shape) == 2:  # single-frame flat
        if darksub:
            log.info("subtracting dark from flat frame")
            diff -= dark
            if dovar:
                diffvar += darkvar
        else:
            log.info("subtracting average bias from flat frame")
            flatbias = get_flatbias(header, pathcal=pathcal)
            log.info("bias level: %f" % flatbias)
            diff -= flatbias

    elif len(data.shape) == 3:  # multi-frame flat
        nframes = data.shape[0]
        if (nframes % 4) == 0:  # 4N-frame flat
            log.info("THIS IS A 4N-FRAME FLAT FILE (ASSUMING CALBOX)")
            mid = nframes // 2
            with catch_warnings():
                simplefilter('ignore')
                low = np.nanmean(diff[mid:, :, :], axis=0)
                high = np.nanmean(diff[:mid, :, :], axis=0)
                diff = high - low
                if dovar:
                    low = diffvar[mid:, :, :]
                    w = np.sum((~np.isnan(low)).astype(int), axis=0)
                    low = np.nansum(low, axis=0)
                    nzi = w > 0
                    low[nzi] /= w[nzi] ** 2
                    high = diffvar[:mid, :, :]
                    w = np.sum((~np.isnan(high)).astype(int), axis=0)
                    high = np.nansum(high, axis=0)
                    nzi = w > 0
                    high[nzi] /= w[nzi] ** 2
                    diffvar = low + high
        else:  # N-frame flat (assume images of same source)
            with catch_warnings():
                simplefilter('ignore')
                diff = np.nanmean(diff, axis=0)
                if dovar:
                    w = np.sum((~np.isnan(diffvar)).astype(int), axis=0)
                    diffvar = np.nansum(diffvar, axis=0)
                    nzi = w > 0
                    diffvar[nzi] /= w[nzi] ** 2
                    diffvar[~nzi] = np.nan
            if darksub:
                log.info("subtracting dark from flat frame")
                diff -= dark
                if dovar:
                    diffvar += darkvar
            else:
                log.info("subtracting average bias from flat frame")
                flatbias = get_flatbias(header, pathcal=pathcal)
                log.info("bias level: %f" % flatbias)
                diff -= flatbias
    else:
        log.error("Invalid flat dimensions %s" % repr(data.shape))
        return

    if normflat:  # normalize
        if ordermask is None:
            ordermask = np.full(data.shape[-2:], True)
        factor = np.nanmedian(diff[ordermask])
        if ~np.isnan(factor) and factor != 0:
            diff /= factor
            # blank out outside order
            diff[~ordermask] = np.nan
            if dovar:
                diffvar /= factor ** 2
                diff[~ordermask] = np.nan
        else:
            log.warning("normalization failed: no valid median")
            addhist(header,
                    "Flat normalization failed (invalid median value)")

    # Kill any zeros generated in the process by hot pixels.
    # The previous version applied a median value across the
    # board.  It was suggested to use maskinterp with a
    # median filter, so that's what we'll do to replace zero
    # values and NaNs
    mask = ~np.isnan(diff)
    mask[mask] &= diff[mask] < 0
    if not mask.all() and mask.any():
        diff = maskinterp(diff, mask=~mask, coplanar=False,
                          func=np.nanmedian, statistical=True, **kwargs)
        if dovar:
            mask = mask & ~np.isnan(diffvar)
            diffvar = maskinterp(diffvar, mask=~mask, coplanar=False,
                                 func=np.nanmedian, statistical=True,
                                 **kwargs)
        # use standard median for any points maskinterp cannot fill
        missing = np.isnan(diff)
        missing[~missing] |= diff[~missing] <= 0
        if missing.any():
            diff[missing] = np.median(diff[~missing])
            if dovar:
                diffvar[missing] = np.median(diffvar[~missing])

    return diff, diffvar


def flatsum(flat, header, flatvar=None, extra=None,
            darkarr=None, darkvar=None, badmap=None,
            imglin=False, ordermask=None,
            jailbar=False, normflat=False, pathcal=None):
    """
    Creates a master flat

    Cleans, jailbar-corrects, droop-corrects, linearity-corrects,
    bias-corrects, averages, and (optionally) normalizes individual
    flat frames to make a single master flat frame.  The bias
    correction depends on the format of the input files:

        - If the number of frames is 4*N, it is assumed that the
          first half of the frames are integrations on a warmer/
          brighter source and the second half of the frames are
          integrations on a cooler/fainter source.  In this case, the
          high count frames will be averaged, the low count frames
          will be averaged, and the low will be subtracted from the
          high.  This will automatically remove any dark current
          or bias; any provided darks will be ignored.
        - Otherwise, it is assumed that all provided flat frames are
          images of the same source.  The frames will be averaged,
          then the provided dark frame will be subtracted.  If no
          dark is provided, an average bias level will be retrieved
          from a look-up table and subtracted from the flat.

    Parameters
    ----------
    flat : numpy.ndarray
        Array containing the flat frames (nframes, nrow, ncol)
    header : astropy.io.fits.header.Header
        The FITS header of the flat file
    darkarr : numpy.ndarray, optional
        Dark array (nrow, ncol) to subtract from the flats array
    darkvar : numpy.ndarray, optional
        Dark variance array (nrow, ncol) to propagate
    flatvar : numpy.ndarray, optional
        Variance array (nflat, nrow, ncol) to update in parallel
        with the flat data array
    extra : dict, optional
        If set, fill with intermediate data products.  keys are
        cleaned, cleanedvar, drooped, droopedvar, imglinearized,
        imglinearizedvar, linearized, linearizedvar.  The var
        suffix signifies variance.  Will be None if variance is
        not propagated.
    badmap : numpy.ndarray, optional
        Bad pixel map (nframe, nrow, ncol).  (0, False)=bad,
        (1, True)=good.  If not set, pixels will not be cleaned
    imglin : bool, optional
        If True, global nonlinearity will be corrected
    ordermask : numpy.ndarray, optional
        Mask indicating edges of orders in spetroscopic flats.
        (1, True)=good, (0, False)=bad pixel.  Areas inside orders
        will be used to normalize the flat; areas outside the orders
        will be set to NaN in the masterflat.
    jailbar : bool, optional
        If True, `jbclean` will be called on the averaged dark frames.
        Should only be set if the JBCLEAN method is 'median';
        otherwise, `jbclean` is called from within `clean`.
    normflat : bool, optional
        If True, the output frame will be normalizes by the median
        of all flat data where ordermask=1.  If unset, the output
        frame will not be normalized.
    pathcal : str, optional
        Path to the data directory.  Used to find the
        default bias leves if darks are not passed.

    Returns
    -------
    2-tuple

    """
    if not isinstance(header, fits.header.Header):
        log.error("invalid header")
        return

    if not isinstance(flat, np.ndarray) or len(flat.shape) not in [2, 3] or \
            np.isnan(flat).all() or np.nanmax(flat) == 0:
        addhist(header, "no flat frames")
        log.error("no flat frames")
        return
    dovar = isinstance(flatvar, np.ndarray) and flatvar.shape == flat.shape
    if not dovar and flatvar is not None:
        msg = "flat variance not propagated (invalid flatvar)"
        addhist(header, msg)
        log.warning(msg)

    if not isinstance(ordermask, np.ndarray):
        ordermask = np.full(flat.shape[-2:], True)
    if extra is None:
        extra = {}

    def valid_copy(n_tuple):
        return tuple(x.copy() if isinstance(x, np.ndarray) else None
                     for x in n_tuple)

    working = valid_copy((flat, flatvar))

    # 1. Clean
    log.info("cleaning flats")
    result = clean_flat(working[0], header, variance=working[1],
                        badmap=badmap, jailbar=jailbar)
    if result is not None:
        working = valid_copy(result)
        extra['cleaned'], extra['cleanedvar'] = working

    # 2. Droop
    log.info("correct droop from flats")
    result = droop(working[0], header, variance=working[1])
    if result is not None:
        working = valid_copy(result)
        extra['drooped'], extra['droopedvar'] = working

    # 3. Image non-linearity
    if imglin:
        log.info("correcting image non-linearity from flats")
        section = read_section(working[0].shape[-1], working[0].shape[-2])
        bglevel = background(working[0], section, mask=ordermask.astype(bool))
        result = imgnonlin(
            working[0], header, siglev=bglevel, variance=working[1])
        if result is not None:
            working = valid_copy(result)
            extra['imglinearized'], extra['imglinearizedvar'] = working

    # 4. Create the master flat
    result = create_master_flat(
        working[0], header, variance=working[1], dark=darkarr,
        darkvar=darkvar, ordermask=ordermask, pathcal=pathcal,
        normflat=normflat)

    if result is None:
        addhist(header, "Could not create master flat")
        log.error("could not create master flat")

    return result
