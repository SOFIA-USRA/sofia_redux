# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import re

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import goodfile, gethdul, hdinsert
from sofia_redux.spectroscopy.smoothres import smoothres
import sofia_redux.instruments.forcast as drip

__all__ = ['clear_model_cache', 'get_model_from_cache',
           'store_model_in_cache', 'get_model']

__model_cache = {}


def clear_model_cache():
    """
    Clear all data from the model cache.
    """
    global __model_cache
    __model_cache = {}


def get_model_from_cache(modelfile, resolution):
    """
    Retrieves model data from the model cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    modelfile : str
        File path to the model file.
    resolution : float
        Spectral resolution used for smoothing.

    Returns
    -------
    filename, wave, unsmoothed, smoothed
        filename : str
            Used to update ATRNFILE in FITS headers.
        wave : numpy.ndarray
            (nwave,) array of wavelengths.
        unsmoothed : numpy.ndarray
            (nwave,) array containing the model data from file
        smoothed : numpy.ndarray
            (nwave,) array containing the smoothed model data
    """
    global __model_cache

    key = modelfile, int(resolution)

    if key not in __model_cache:
        return

    if not goodfile(modelfile):
        try:
            del __model_cache[key]
        except KeyError:   # pragma: no cover
            # could happen in race conditions, working in parallel
            pass
        return

    modtime = str(os.path.getmtime(modelfile))
    if modtime not in __model_cache.get(key, {}):
        return

    log.debug("Retrieving model data from cache (%s, resolution %s) " % key)
    return __model_cache.get(key, {}).get(modtime)


def store_model_in_cache(modelfile, resolution, filename, wave,
                         unsmoothed, smoothed):
    """
    Store model data in the model cache.

    Parameters
    ----------
    modelfile : str
        File path to the model file
    resolution : float
        Spectral resolution used for smoothing.
    filename : str
        Used to update ATRNFILE in FITS headers.
    wave : numpy.ndarray
        (nwave,) array of wavelengths.
    unsmoothed : numpy.ndarray
        (nwave,) array containing the model data from file
    smoothed : numpy.ndarray
        (nwave,) array containing the smoothed model data

    """
    global __model_cache
    key = modelfile, int(resolution)
    log.debug("Storing model data in cache (%s, resolution %s)" % key)
    __model_cache[key] = {}
    modtime = str(os.path.getmtime(modelfile))
    __model_cache[key][modtime] = (
        filename, wave, unsmoothed, smoothed)


def get_model(header, resolution, filename=None,
              get_unsmoothed=False, model_dir=None):
    """
    Retrieve reference standard model data.

    Model files in the data/model_files directory should be named
    according to the object and date to which they apply, as:

        [object]_[YYYYMMDD(HH)]_model.fits

    The date may be omitted, if the model applies to all dates.
    For example, the file generated for Ceres on Jan. 14 2020
    should be named:

        ceres_20200114_model.fits

    The procedure is:

        1. Identify model file by object name and date.
        2. Read model data from file and smooth to expected spectral
           resolution.
        3. Store in cache and return data array.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        ATRNFILE keyword is written to the provided FITS header,
        containing the name of the model file used.
    resolution : float
        Spectral resolution to which model data should be smoothed.
    filename : str, optional
        Model file to be used.  If not provided,
        a default file will be retrieved from the data/grism/standard_models
        directory.  The file with a matching object name and the closest
        date will be used.
    get_unsmoothed : bool, optional
        If True, return the unsmoothed model data with original
        wavelength array in addition to the smoothed array.
    model_dir : str, optional
        Path to a directory containing model reference FITS files.
        If not provided, the default set of files packaged with the
        pipeline will be used.

    Returns
    -------
    model : numpy.ndarray
        A (2, nw) array containing wavelengths and model data.
    unsmoothed : numpy.ndarray, optional
        A (2, nw) array containing wavelengths and unsmoothed
        model data, returned only if get_unsmoothed is set.
    """
    if filename is not None:
        if not goodfile(filename, verbose=True):
            log.warning('File {} not found; '
                        'retrieving default'.format(filename))
            filename = None
    if filename is None:
        if not isinstance(header, fits.header.Header):
            log.error("Invalid header")
            return

        # get standardized object name
        objname = str(header.get('OBJECT', 'UNKNOWN')).lower()
        objname = re.sub(r'[ _\-@]|jpl', '', objname)

        # get YYYYMMDDHH date number
        date = header.get('DATE-OBS', '').replace('T', ' ')
        date = date.replace('-', ' ').replace(':', ' ')
        date = date.split()
        dateobs = 9999999999
        if len(date) >= 4:
            try:
                dateobs = int(''.join(date[0:4]))
            except ValueError:
                pass
        elif len(date) >= 3:
            try:
                dateobs = int(''.join(date[0:3]) + '00')
            except ValueError:
                pass

        log.debug('Object, date: {} {}'.format(objname, dateobs))

        if model_dir is not None:
            if not os.path.isdir(str(model_dir)):
                log.warning('Cannot find model directory: %s' % model_dir)
                log.warning('Using default model set.')
                model_dir = None
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(drip.__file__),
                                     'data', 'grism', 'standard_models')

        model_files = glob.glob(os.path.join(model_dir, '*fits'))
        regex = re.compile(r'^([0-9A-Za-z]+)_?([0-9]{8,10})?.*\.fits$')

        minval = None
        for f in model_files:
            match = regex.match(os.path.basename(f))
            if match is not None:
                stdname = str(match.group(1)).lower()

                # match if standard name is contained in object name
                if stdname not in objname:
                    continue

                if match.group(2) is None:
                    stddate = dateobs
                else:
                    stddate = match.group(2)
                    if len(stddate) < 10:
                        stddate = int(stddate[:9] + '00')
                    else:
                        stddate = int(stddate[:11])

                # keep closest date
                val = np.abs(dateobs - stddate)
                if minval is None or val < minval:
                    minval, filename = val, f

    if not isinstance(filename, str):
        log.error("No model file found")
        return
    log.debug("Using model file %s" % filename)

    # Read the model data from cache if possible
    modeldata = get_model_from_cache(filename, resolution)
    if modeldata is not None:
        modelfile, wave, unsmoothed, smoothed = modeldata
    else:
        hdul = gethdul(filename, verbose=True)
        if hdul is None or hdul[0].data is None:
            log.error("Invalid data in model file %s" % filename)
            return
        data = hdul[0].data
        wave = data[0]
        unsmoothed = data[1]
        smoothed = smoothres(data[0], data[1], resolution)
        modelfile = os.path.basename(filename)
        store_model_in_cache(filename, resolution, modelfile,
                             data[0], data[1], smoothed)

    hdinsert(header, 'MODLFILE', modelfile)

    if not get_unsmoothed:
        return np.vstack((wave, smoothed))
    else:
        return np.vstack((wave, smoothed)), np.vstack((wave, unsmoothed))
