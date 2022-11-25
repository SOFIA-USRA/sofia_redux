# Licensed under a 3-clause BSD style license - see LICENSE.rst

import re

from astropy import log
from astropy.io import fits
from astropy.time import Time
import numpy as np

from sofia_redux.toolkit.utilities.fits import getdata
from sofia_redux.toolkit.utilities.func import goodfile

__all__ = ['get_detsec', 'check_data_dimensions',
           'check_variance_dimensions', 'get_reset_dark',
           'set_elapsed_time', 'parse_central_wavenumber']


def get_detsec(header_or_str):
    """
    Parse the DETSEC keyword in a header or from a string.

    DETSEC should be a string of the form "[xstart,xstop,ystart,ystop]"
    where each entry marks the beginning and end indices of a 2D.
    array (ny, nx).

    Parameters
    ----------
    header_or_str : fits.Header or str
        Either a header containing the DETSEC keyword/value or a string
        containing the value.

    Returns
    -------
    xstart, xstop, ystart, ystop : int, int, int, int
        The start and stop indices marking an area of a 2D array.
    """
    if isinstance(header_or_str, fits.Header):
        detsec = header_or_str.get('DETSEC', 'UNKNOWN')
        nx = header_or_str.get('NSPAT')
        ny = header_or_str.get('NSPEC')
    else:
        detsec = header_or_str
        nx, ny = None, None

    if not isinstance(detsec, str):
        raise TypeError("Must supply a FITS header or string")

    if detsec == 'UNKNOWN' and None not in [nx, ny]:
        xstart = 0
        xstop = nx
        ystart = 0
        ystop = ny
    else:
        try:
            detsec = [int(x) for x in re.split(r'[\[\]\s:,]', detsec)
                      if x != '']
        except ValueError:
            raise ValueError("DETSEC must be of the format [#,#,#,#]")
        if len(detsec) != 4:
            raise ValueError("DETSEC must be of the format [#,#,#,#]")
        xstart, xstop, ystart, ystop = detsec
        xstart -= 1
        ystart -= 1

    if xstart < 0 or ystart < 0:
        raise ValueError(
            "Starting indices for DETSEC in x and y must be positive nonzero")

    return xstart, xstop, ystart, ystop


def check_data_dimensions(**kwargs):
    """
    Check the data dimensions and return number of frames.

    Input data should be 2 or 3 dimensions, matching specified
    x and y sizes.

    Parameters
    ----------
    kwargs : dict
        May contain a 'params' key, containing a dict with 'data',
        'nx', 'ny' input.  Otherwise, it may contain 'data', 'nx',
        and 'ny' directly specified.  The 'data' key should contain
        an array; 'nx' and 'ny' should specify its expected x and y
        dimensions, respectively.

    Returns
    -------
    nz : int
        The number of frames in the input data cube. Returns 1 if
        the input has 2 dimensions.
    """
    params = kwargs.get('params')
    if params is None:
        data = kwargs.get('data')
        nx = kwargs.get('nx')
        ny = kwargs.get('ny')
    else:
        data = params['data']
        nx = params['nx']
        ny = params['ny']
    if data.ndim <= 2:
        nz = 1
    else:
        nz = data.shape[0]
    if data.ndim == 2:
        check = data.shape != (ny, nx)
    elif data.ndim == 3:
        check = data.shape != (nz, ny, nx)
    else:
        check = True
    if check:
        raise RuntimeError
    return nz


def check_variance_dimensions(variance, nx, ny, nz):
    """
    Check variance dimensions for expected shape.

    Parameters
    ----------
    variance : numpy.ndarray
        2D or 3D variance array to check.
    nx : int
        Expected x dimension.
    ny : int
        Expected y dimension.
    nz : int
        Expected z dimension (number of frames).

    Returns
    -------
    valid : bool
        True if variance matches expected dimensions.
    """
    if variance is None:
        return False
    if variance.ndim <= 2:
        nvz = 1
    else:
        nvz = variance.shape[0]
    if variance.ndim == 2:
        check = variance.shape != (ny, nx)
    elif variance.ndim == 3:
        check = variance.shape != (nz, ny, nx)
    else:
        check = True
    if check or nvz != nz:
        raise RuntimeError
    return True


def get_reset_dark(header):
    """
    Get a reset dark image from a file on disk.

    Dark files are expected to be single-extension FITS images
    containing raw dark data. If the input file contains multiple frames,
    the first one is returned as the reset dark.

    If a detector section is specified in the input header, the
    corresponding section is extracted from the full dark image.

    Parameters
    ----------
    header : fits.Header
        FITS header containing a DRKFILE key, specifying the dark
        file to read.

    Returns
    -------
    dark1s : numpy.ndarray
        The 2D dark image.
    """
    darkfile = str(header.get('DRKFILE', 'UNKNOWN'))
    if not goodfile(darkfile, verbose=False):
        raise ValueError(f'Cannot open dark file {darkfile}')
    log.info(f'Using reset dark file {darkfile}')
    dark = getdata(darkfile)

    # Take the shortest dark frame as the 'bias'
    xstart, xstop, ystart, ystop = get_detsec(header)
    try:
        if dark.ndim == 2:
            dark1s = dark[ystart:ystop, xstart:xstop]
        else:
            dark1s = dark[0, ystart:ystop, xstart:xstop]
    except (ValueError, IndexError):
        raise ValueError(f"Dark file has wrong dimensions "
                         f"{repr(dark.shape)}.") from None

    return dark1s


def set_elapsed_time(header):
    """
    Set the TOTTIME key in the header to the total elapsed time.

    Uses UTCSTART, UTCEND, and DATE-OBS keywords.

    Parameters
    ----------
    header : fits.Header
        Header to update.
    """
    # start/end is time only
    utcstart = header.get('UTCSTART', 'UNKNOWN')
    utcend = header.get('UTCEND', 'UNKNOWN')

    # get date from date-obs
    date = header.get('DATE-OBS', 'UNKNOWN')

    try:
        ymd = date.split('T')[0]
        start_time = Time(f'{ymd}T{utcstart}', scale='utc', format='isot').unix
        end_time = Time(f'{ymd}T{utcend}', scale='utc', format='isot').unix
        tottime = float(end_time - start_time)
    except (ValueError, TypeError):
        log.warning('UTCSTART/END not understood')
        return

    if tottime > 0:
        header['TOTTIME'] = tottime
    else:
        log.warning('UTCSTART/END not understood')
        return


def parse_central_wavenumber(header):
    """
    Parse the central wavenumber from the input header.

    Typically, the WAVENO0 keyword is set in the raw data by
    the instrument control software according to the planned
    observation settings. However, it is usually insufficiently
    accurate for wavelength calibration purposes, so it may
    be overridden by a more accurate value in the WNO0 keyword.

    If WNO0 is set to a float value greater than zero, it
    is used as the central wavenumber.  If not, WAVENO0 is
    used as the central wavenumber.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The FITS header containing wavenumber keys.

    Returns
    -------
    central_waveno : float
        The wavenumber determined from the input header.
    """
    waveno0 = header.get('WAVENO0')
    wno0 = header.get('WNO0')
    try:
        wno0 = float(wno0)
    except (ValueError, TypeError):
        wnoc = waveno0
    else:
        if wno0 == -9999 or wno0 <= 0 and waveno0 is not None:
            wnoc = waveno0
        else:
            wnoc = np.abs(wno0)
    return wnoc
