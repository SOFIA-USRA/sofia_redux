# Licensed under a 3-clause BSD style license - see LICENSE.rst

from datetime import datetime
import os

from astropy import log
from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities import hdinsert, goodfile

__all__ = ['clear_badpix_cache', 'get_badpix_from_cache',
           'store_badpix_in_cache', 'read_defaults_table',
           'get_badpix']

__badpix_cache = {}


def clear_badpix_cache():
    """
    Clear all data from the badpix cache.
    """
    global __badpix_cache
    __badpix_cache = {}


def get_badpix_from_cache(badpixfile):
    """
    Retrieves bad pixel masks or default file from the badpix cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    badpixfile : str
        File path to the badpix file

    Returns
    -------
    badpix : numpy.ndarray or pandas.DataFrame
        Bad pixel mask or defaults table
    """
    global __badpix_cache

    if badpixfile not in __badpix_cache:
        return

    if not goodfile(badpixfile):
        try:
            del __badpix_cache[badpixfile]
        except KeyError:   # pragma: no cover
            # could happen in race conditions in parallel processing
            pass
        return

    modtime = str(os.path.getmtime(badpixfile))
    if modtime not in __badpix_cache.get(badpixfile, {}):
        return

    log.debug("Retrieving data from badpix file (%s) in cache" % badpixfile)
    return __badpix_cache.get(badpixfile, {}).get(modtime)


def store_badpix_in_cache(badpixfile, badpix):
    """
    Store badpix data in the badpix cache.

    Parameters
    ----------
    badpixfile : str
        File path to the badpix file
    badpix : numpy.ndarray or pandas.DataFrame
        Bad pixel mask or defaults table
    """
    global __badpix_cache
    log.debug("Storing data from badpix (%s) in cache" % badpixfile)
    __badpix_cache[badpixfile] = {}
    modtime = str(os.path.getmtime(badpixfile))
    __badpix_cache[badpixfile][modtime] = badpix


def read_defaults_table():
    """
    Read the badpix defaults table.
    """

    datapath = os.path.join(os.path.dirname(fifi_ls.__file__), 'data')
    default_file = os.path.join(
        datapath, 'badpix_files', 'badpix_default.txt')

    defaults_table = get_badpix_from_cache(default_file)
    if defaults_table is not None:
        return defaults_table

    log.debug("Reading badpix defaults file")
    if not goodfile(default_file, verbose=True):
        msg = "Could not read badpix default file: {}".format(default_file)
        log.error(msg)
        raise ValueError(msg)

    table = pandas.read_csv(
        default_file, comment='#', delim_whitespace=True,
        names=['date', 'channel', 'filename'],
        dtype={'date': int},
        converters={'filename': lambda x: os.path.join(datapath, x),
                    'channel': lambda x: str(x).upper().strip()})

    store_badpix_in_cache(default_file, table)
    return table


def get_badpix(header, filename=None):
    """
    Retrieve bad pixel data.

    Badpix files are identified by a configuration file called
    badpix_default.txt in the data/badpix_files directory.  This
    file must contain 3 columns: end date (YYYYMMDD), channel
    (b1, b2, or r), and filepath (relative to the data directory).

    The procedure is:

        1. Identify badpix file, unless override is provided
        2. Read badpix data from file
        3. Return badpix array

    Parameters
    ----------
    header : fits.Header
        FITS header from which to determine badpix file.  Will be
        updated with BDPXFILE keyword.
    filename : str, optional
        Direct path to a badpix file (overrides header logic)

    Returns
    -------
    numpy.ndarray
        (n_values, (spexel number, spaxel_number))
    """
    if isinstance(filename, str) and goodfile(filename, verbose=True):
        badpix_file = filename[:]
    else:
        if not isinstance(header, fits.Header):
            log.error("Invalid header")
            return

        dateobs = header.get(
            'DATE-OBS', Time(datetime.utcnow(), format='datetime').isot)
        try:
            yyyymmdd = int(dateobs.split('T')[0].replace('-', ''))
        except (ValueError, IndexError):
            log.warning(
                "Could not determine DATE-OBS - using most recent file")
            yyyymmdd = 88888888
        channel = header.get('CHANNEL', 'UNKNOWN').strip().upper()
        channel = 'B' if channel == 'BLUE' else 'R'

        table = read_defaults_table()
        badpix_file = table.loc[table[
            (table['channel'] == channel)
            & (table['date'] >= yyyymmdd)]['date'].idxmin()].filename

    log.debug("Using badpix file %s" % badpix_file)
    data = get_badpix_from_cache(badpix_file)
    if data is not None:
        if isinstance(header, fits.Header):
            hdinsert(header, 'BDPXFILE', os.path.basename(badpix_file))
        return data

    log.debug("Loading badpix file %s" % badpix_file)

    if isinstance(header, fits.Header):
        hdinsert(header, 'BDPXFILE', os.path.basename(badpix_file))

    try:
        data = pandas.read_csv(
            badpix_file, names=['spaxel', 'spexel'],
            dtype={'spaxel': int, 'spexel': int},
            comment='#', delim_whitespace=True).values
    except ValueError:
        msg = "Invalid badpix file: {}".format(badpix_file)
        log.error(msg)
        return

    if isinstance(data, np.ndarray) and len(data.shape) == 2:
        # Convert coordinates to 0-based array coordinates
        data -= 1

    # flipping the axis here as it's used for indexing a numpy array
    # (y, x) ordering, unlike file format
    data = np.flip(data, axis=1)
    store_badpix_in_cache(badpix_file, data)

    return data
