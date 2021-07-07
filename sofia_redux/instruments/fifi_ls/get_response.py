# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import numpy as np
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities import goodfile, hdinsert, gethdul

__all__ = ['clear_response_cache', 'get_response_from_cache',
           'store_response_in_cache', 'get_response']

__response_cache = {}


def clear_response_cache():
    """
    Clear all data from the response cache.
    """
    global __response_cache
    __response_cache = {}


def get_response_from_cache(responsefile):
    """
    Retrieves response data from the response cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    responsefile : str
        File path to the response file

    Returns
    -------
    response : numpy.ndarray
    """
    global __response_cache

    if responsefile not in __response_cache:
        return

    if not goodfile(responsefile):
        try:
            del __response_cache[responsefile]
        except KeyError:   # pragma: no cover
            pass
        return

    modtime = str(os.path.getmtime(responsefile))
    if modtime not in __response_cache.get(responsefile, {}):
        return

    log.debug("Retrieving data from response file (%s) in cache" %
              responsefile)
    return __response_cache.get(responsefile, {}).get(modtime)


def store_response_in_cache(responsefile, response):
    """
    Store response data in the response cache

    Parameters
    ----------
    responsefile : str
        File path to the response file
    response : numpy.ndarray
        Response data.
    """
    global __response_cache
    log.debug("Storing data from flats (%s) in cache" % responsefile)
    __response_cache[responsefile] = {}
    modtime = str(os.path.getmtime(responsefile))
    __response_cache[responsefile][modtime] = response


def get_response(header, filename=None):
    """
    Retrieve instrumental response data.

    Reponse files are identified by a configuration file called
    response_default.txt in the data/response_files directory.  This
    file must contain 4 columns: end date (YYYYMMDD), channel (b1,
    b2, or r), dichroic (105 or 130), and filepath (relative to the
    data directory).

    The procedure is:
        1. Identify response file, unless override is provided.
        2. Read response data from file.
        3. Return response array.

    Parameters
    ----------
    header : fits.Header
        FITS header for input data to match.  The header will be updated
        with the RSPNFILE keyword containing the name of the response
        file used.
    filename : str, optional
        Response file to be used.  If not provided, a default file will
        be retrieved from the data/response files directory, matching
        the date, channel, and dichroic of the input header.  If an
        override file is provided, it should be a FITS image file
        containing wavelength, response, and error on the response in
        an array of three rows in the primary extension.

    Returns
    -------
    numpy.ndarray
        (3, nw) array where [0, :] = wavelength, [1, :] = response data,
        and [2, :] = error.
    """
    if not isinstance(header, fits.Header):
        log.error("Invalid header")
        return
    if filename is not None:
        if not goodfile(filename, verbose=True):
            log.error('Could not find file: {}.  '
                      'Using default.'.format(filename))
            filename = None
    if filename is None:
        datapath = os.path.join(os.path.dirname(fifi_ls.__file__), 'data')
        default_file = os.path.join(
            datapath, 'response_files', 'response_default.txt')
        if not goodfile(default_file, verbose=True):
            log.error('Cannot read response default file')
            return
        required_keywords = ['CHANNEL', 'G_ORD_B', 'DICHROIC']
        for key in required_keywords:
            if key not in header:
                log.error("Header missing %s keyword" % key)
                return

        obsdate = header.get('DATE-OBS', '9999-99-99')
        try:
            obsdate = int(''.join([x for x in obsdate[:10].split('-')]))
        except (TypeError, ValueError):
            obsdate = 99999999
        channel = str(header['CHANNEL']).upper().strip()
        dichroic = str(header['DICHROIC'])
        b_order = str(header['G_ORD_B'])

        # also check for the order filter for blue
        # This keyword is only present from 10/2019 on.
        # If not present, or it's a bad value, assume the value
        # matches the order
        if 'G_FLT_B' in header:
            b_filter = str(header['G_FLT_B']).upper().strip()
            if b_filter not in ['1', '2']:
                b_filter = b_order
        else:
            b_filter = b_order

        if channel == 'BLUE':
            if b_order in ['1', '2']:
                if b_filter == b_order:
                    chan = 'b%s' % b_order
                else:
                    chan = 'b%s%s' % (b_order, b_filter)
            else:
                log.error("Invalid blue grating order")
                return
        else:
            chan = 'r'

        try:
            df = pandas.read_csv(
                default_file, comment='#', delim_whitespace=True,
                names=['date', 'chan', 'dichroic', 'filename'])
            df = df.sort_values('date')
            rows = df[(df['date'] >= obsdate)
                      & (df['chan'] == chan)
                      & (df['dichroic'].apply(str) == dichroic)]
        except (ValueError, TypeError):
            log.error('Cannot read response default file')
            return
        if len(rows) == 0:
            log.error("Unable to find response file for %s, %s, %s" %
                      (obsdate, chan, dichroic))
            return
        filename = os.path.join(datapath, rows.iloc[0]['filename'])

    response = get_response_from_cache(filename)
    if response is not None:
        hdinsert(header, 'RSPNFILE', os.path.basename(filename))
        return response

    hdul = gethdul(filename, verbose=True)
    if hdul is None or hdul[0].data is None:
        log.error("Invalid data in response file %s" % filename)
        return

    response = np.asarray(hdul[0].data, dtype=float)
    hdinsert(header, 'RSPNFILE', os.path.basename(filename))
    store_response_in_cache(filename, response)

    return response
