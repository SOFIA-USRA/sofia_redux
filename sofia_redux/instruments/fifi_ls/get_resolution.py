# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities import goodfile, hdinsert

__all__ = ['clear_resolution_cache', 'get_resolution_from_cache',
           'store_resolution_in_cache', 'get_resolution']

__resolution_cache = {}


def clear_resolution_cache():
    """
    Clear all data from the resolution cache.
    """
    global __resolution_cache
    __resolution_cache = {}


def get_resolution_from_cache(resfile):
    """
    Retrieves table from the resolution cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    resfile : str
        File path to the resolution file

    Returns
    -------
    resolution : pandas.DataFrame
        Resolution table
    """
    global __resolution_cache

    if resfile not in __resolution_cache:
        return

    if not goodfile(resfile):
        try:
            del __resolution_cache[resfile]
        except KeyError:   # pragma: no cover
            pass
        return

    modtime = str(os.path.getmtime(resfile))
    if modtime not in __resolution_cache.get(resfile, {}):
        return

    log.debug("Retrieving data from resolution file (%s) in cache" %
              resfile)
    return __resolution_cache.get(resfile, {}).get(modtime)


def store_resolution_in_cache(resfile, resolution):
    """
    Store resolution data in the resolution cache.

    Parameters
    ----------
    resfile : str
        File path to the resolution file
    resolution : pandas.DataFrame
        Resolution table
    """
    global __resolution_cache
    log.debug("Storing data from resolution (%s) in cache" % resfile)
    __resolution_cache[resfile] = {}
    modtime = str(os.path.getmtime(resfile))
    __resolution_cache[resfile][modtime] = resolution


def get_resolution(header, wmean=None, spatial=False):
    """
    Retrieve expected spectral or spatial resolution.

    Requires spectral_resolution.txt file in fifi_ls/data/resolution.
    This file must have 4 columns: channel (b1, b2, or r), central
    wavelength (um), spectral resolution (um), and spatial resolution
    (FWHM, in arcsec).

    The header will be updated with the 'RESFILE' keyword specifying
    the name of the resolution file used.

    The procedure is:

        1. Read resolution data from spectral_resolution.txt configuration
           file.
        2. Match resolution to input data, using keywords CHANNEL, G_ORD_B,
           and G_WAVE_R or G_WAVE_B.
        3. Return spectral or spatial resolution.

    Parameters
    ----------
    header : fits.Header
    wmean : float, optional
        If set, use as the central wavelength for the data.  If
        not set, will use keyword G_WAVE_B or G_WAVE_R as appropriate
        for the data.
    spatial : bool, optional
        If True, will return spatial resolution instead of spectral
        resolution

    Returns
    -------
    float
        Expected resolution  in um for spectral resolution and arcsec
        for spatial
    """
    if not isinstance(header, fits.Header):
        msg = 'Invalid header'
        log.error(msg)
        raise ValueError(msg)

    resfile = os.path.join(os.path.dirname(fifi_ls.__file__), 'data',
                           'resolution', 'spectral_resolution.txt')

    if not goodfile(resfile, verbose=True):
        msg = "Cannot read resolution file: %s" % resfile
        log.error(msg)
        raise ValueError(msg)

    required_keywords = ['CHANNEL', 'G_ORD_B', 'G_WAVE_R', 'G_WAVE_B']
    for key in required_keywords:
        if key not in header:
            msg = "Header missing %s keyword" % key
            log.error(msg)
            raise ValueError(msg)

    channel = header['CHANNEL']
    b_order = str(header['G_ORD_B']).strip()
    if channel == 'BLUE':
        if b_order not in ['1', '2']:
            msg = "Invalid blue grating order (%s)" % b_order
            log.error(msg)
            raise ValueError(msg)
        ch = 'b%s' % b_order
    elif channel == 'RED':
        ch = 'r'
    else:
        ch = 'unknown'

    if ch == 'unknown':
        log.warning('Channel is unknown; setting resolution to default values')
        hdinsert(header, 'RESFILE', os.path.basename(resfile))
        return 5.0 if spatial else 1000.0

    # Get central wavelength
    if wmean is None:
        wmean = header['G_WAVE_R'] if ch == 'r' else header['G_WAVE_B']
    try:
        wmean = float(wmean)
    except (TypeError, ValueError):
        msg = "Invalid wavelength mean: %s" % repr(wmean)
        log.error(msg)
        raise ValueError(msg)

    hdinsert(header, 'RESFILE', os.path.basename(resfile))

    # Find appropriate resolution
    df = get_resolution_from_cache(resfile)
    if df is None:
        log.debug("Loading resolution file: %s" % resfile)
        names = ['chan', 'wavelength', 'res', 'mfwhm']
        df = pandas.read_csv(resfile, comment='#', names=names,
                             delim_whitespace=True)
        store_resolution_in_cache(resfile, df)

    col = 'mfwhm' if spatial else 'res'
    resolution = df.loc[
        (df[df['chan'] == ch].wavelength - wmean).abs().idxmin()][col]

    return float(resolution)
