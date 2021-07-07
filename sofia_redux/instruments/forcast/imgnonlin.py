# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re

from astropy import log
from astropy.io import fits
import numpy as np
from numpy.polynomial.polynomial import polyval

from sofia_redux.toolkit.utilities.fits import add_history_wrap, hdinsert, kref

from sofia_redux.instruments.forcast.getdetchan import getdetchan
from sofia_redux.instruments.forcast.getpar import getpar

addhist = add_history_wrap('Image nonlin')

__all__ = ['get_siglev', 'get_camera_and_capacitance',
           'get_reference_scale', 'get_coefficients',
           'get_coeff_limits', 'imgnonlin']


def get_siglev(header):
    """
    Return the signal level from the header

    Parameters
    ----------
    header : astropy.io.fits.header.Header

    Returns
    -------
    numpy.ndarray

    """
    siglev = getpar(header, 'NLINSLEV', dtype=str, default='NONE', warn=True,
                    comment='Signal level of background')
    if siglev == 'NONE':
        return
    siglev = [float(x) for x in re.sub(r'[\[\]]', '', siglev).split(',')]
    return np.array(siglev)


def get_camera_and_capacitance(header):
    """
    Read header and determine camera

    Parameters
    ----------
    header : astropy.io.fits.header.Header

    Returns
    -------
    str
       camera + capacitance.  All uppercase.  For example, 'LWCLO'
    """
    detchan = getdetchan(header)
    camera = 'SWC' if detchan == 'SW' else 'LWC'
    epadu = getpar(header, 'EPERADU', dtype=int, default=None, warn=True)
    if epadu is None:
        return
    cap = {136: 'Lo', 1294: 'Hi'}.get(epadu)
    if cap is None:
        return
    return (camera + cap).upper().strip()


def get_reference_scale(header, camcap, update=False):
    """
    Get non-linearity scale from the header based on camera and capacitance

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    camcap : str
        camera + 2-letter-capacitance, e.g., SWCHI, LWCLO, etc.
    update : bool, optional
        If True, update the header with a HISTORY message stating the
        scale

    Returns
    -------
    tuple of float
        reference, scale
    """
    refsig = getpar(header, 'NLR' + camcap,
                    dtype=float, default=9000, warn=True,
                    comment='count reference for linearity correction')
    scale = getpar(header, 'NLS' + camcap,
                   dtype=float, default=refsig, warn=True,
                   comment='count scale for linearity correction')
    if update:
        addhist(header, 'Scale is %f' % scale)

    return refsig, scale


def get_coefficients(header, camcap, update=False):
    """
    Get non-linearity coefficients from the header based on camera and
    capacitance

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    camcap : str
        camera + 2-letter-capacitance, e.g., SWCHI, LWCLO, etc.
    update : bool, optional
        If True, update the header with the coefficients.  keys will
        be NLINC# where # represents a number.  The NLC + camap key
        will also be removed as it does not fit in the line.  A
        HISTORY message will also be appended stating the read
        coefficients

    Returns
    -------
    numpy.ndarray
        float coefficient values from header
    """
    # Get coeff depending on camera and cap
    cread = getpar(header, 'NLC' + camcap, dtype=str, default='NONE',
                   warn=True, comment='linearity correction coefficients')
    if cread == 'NONE':
        return
    coeffs = [float(x) for x in re.sub(r'[\[\]]', '', cread).split(',')]

    if update:
        addhist(header, 'Coeff=%s' % cread)
        key = ('NLC' + camcap)[:8]
        if key in header:
            del header[key]
        for idx, val in enumerate(coeffs):
            hdinsert(header, 'NLINC%s' % idx, val, refkey=kref,
                     comment='linearity correction coeff #%s' % idx)

    return np.array(coeffs)


def get_coeff_limits(header, camcap, update=False):
    """
    Get non-linearity coefficient limits from the header based on camera
    and capacitance

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    camcap : str
        camera + 2-letter-capacitance, e.g., SWCHI, LWCLO, etc.
    update : bool, optional
        If True, update the header with HISTORY messages

    Returns
    -------
    numpy.ndarray
        float coefficient limit values from header of length 2
    """
    # Get lims depending on camera and cap
    limread = getpar(header, 'LIM' + camcap, dtype=str, default='NONE',
                     comment='linearity correction limits', warn=True)
    if limread == 'NONE':
        return
    if update:
        addhist(header, 'level limits are %s' % limread)

    lims = [float(x) for x in re.sub(r'[\[\]]', '', limread).split(',')]
    if len(lims) != 2:
        return
    return np.array(lims)


def imgnonlin(data, header, siglev=None, variance=None):
    """
    Corrects for non-linearity in detector response do to general background

    The header must contain the information to determine the camera.  If
    siglev is not passed, the header must also contain a keyword
    (NLINSLEV) to indicate the background level and size of the section
    used to calculate the level.  In practice, this means that
    sofia_redux.instruments.forcast.background should be run first to
    calculate the background level.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (nimage, nrow, ncol)
    header : astropy.io.fits.header.Header
        Input FITS header.  Will be updated with a HISTORY message
    siglev : array_like, optional
        Background level.  There should be a single value for each input
        frame
    variance : numpy.ndarray, optional
        Variance array (nimage, nrow, ncol) to update in parallel with the
        data frame.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The linearity corrected array (nimage, nrow, ncol) or (nrow, ncol)
        The propagated variance array (nimage, nrow, ncol) or (nrow, ncol)
    """
    if not isinstance(header, fits.header.Header):
        log.error("invalid header")
        return

    if not isinstance(data, np.ndarray) or len(data.shape) not in [2, 3]:
        addhist(header, 'not corrected (invalid data)')
        log.error("invalid data")
        return
    ndim = len(data.shape)
    d = data.copy()
    if ndim == 2:
        d = np.array([d])

    dovar = variance is not None and variance.shape == data.shape
    var = variance.copy() if dovar else None
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (invalid variance)')
        log.error('invalid variance')
    if ndim == 2:
        var = np.array([var])
    elif not dovar:
        var = [None] * 3

    # Read background level in header
    if siglev is None:
        siglev = get_siglev(header)
    else:
        siglev = np.array(siglev)
    if not isinstance(siglev, np.ndarray):
        addhist(header, 'not corrected (invalid signal levels)')
        log.error('invalid signal levels')
        return
    elif len(siglev) != d.shape[0]:
        addhist(header, 'not corrected (mismatch data and signal levels)')
        log.error('signal size does not match data shape')
        return

    # Get the camera and capacitance level
    camcap = get_camera_and_capacitance(header)
    if camcap is None:
        addhist(header, 'not corrected (unknown capacitance)')
        log.error('E/ADU is not correctly defined')
        return
    log.info("Using camera %s with %s capacitance" % (camcap[:3], camcap[3:]))

    # get non-linearity reference and scale
    refscale = get_reference_scale(header, camcap, update=True)

    # get non-linearity coefficients
    coeffs = get_coefficients(header, camcap, update=True)
    if coeffs is None:
        addhist(header, 'not corrected (invalid non-linearity coefficients)')
        log.error('invalid non-linearity coefficients')
        return

    # get non-linearity coefficient limits
    lims = get_coeff_limits(header, camcap, update=True)
    if lims is None:
        msg = 'invalid limits of non-linearity coefficients'
        addhist(header, 'not corrected (%s)' % msg)
        log.error(msg)
        return

    # Check limits
    outside = np.where((siglev < lims[0]) | (siglev > lims[1]))[0]
    if len(outside) > 0:
        hdinsert(header, 'NLINFLAG', True, refkey=kref,
                 comment='flag outside range levels in lin. correction')
        for idx in outside:
            msg = 'Signal level %s for plane %i outside range' % (
                siglev[idx], idx)
            addhist(header, msg)
            log.warning("level %f outside fit range: %s" % (siglev[idx], lims))
        addhist(header, 'not corrected (level outside correction range)')
        log.warning('level outside correction range')
        return

    # Apply correction
    plane = 0
    xval = (siglev - refscale[0]) / refscale[1]
    corr = polyval(xval, coeffs)
    for factor, frame, vframe in zip(corr, d, var):
        msg = 'factor of plane %s is %s' % (plane, factor)
        log.info(msg)
        addhist(header, msg)
        frame /= factor
        if dovar:
            vframe /= factor ** 2
        plane += 1

    if ndim == 2:
        d, var = d[0], var[0]
    elif not dovar:
        var = None

    return d, var
