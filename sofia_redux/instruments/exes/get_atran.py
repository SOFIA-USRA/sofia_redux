# Licensed under a 3-clause BSD style license - see LICENSE.rst

import glob
import os
import re

from astropy import log, units
from astropy.io import fits
import numpy as np

from sofia_redux.instruments import exes
from sofia_redux.toolkit.utilities.fits import goodfile, gethdul, hdinsert
from sofia_redux.spectroscopy.smoothres import smoothres

__all__ = ['clear_atran_cache', 'get_atran_from_cache',
           'store_atran_in_cache', 'get_atran']

__atran_cache = {}


def clear_atran_cache():
    """
    Clear all data from the atran cache.
    """
    global __atran_cache
    __atran_cache = {}


def get_atran_from_cache(atranfile, resolution):
    """
    Retrieves atmospheric transmission data from the atran cache.

    Checks to see if the file still exists, can be read, and has not
    been altered since last time.  If the file has changed, it will be
    deleted from the cache so that it can be re-read.

    Parameters
    ----------
    atranfile : str
        File path to the atran file
    resolution : float
        Spectral resolution used for smoothing.

    Returns
    -------
    tuple
        filename : str
            Used to update ATRNFILE in FITS headers.
        wave : numpy.ndarray
            (nwave,) array of wavelengths.
        unsmoothed : numpy.ndarray
            2D array containing the atran data from file
        smoothed : numpy.ndarray
            (nwave,) array containing the smoothed atran data
    """
    global __atran_cache

    key = atranfile, int(resolution)

    if key not in __atran_cache:
        return

    if not goodfile(atranfile):
        try:
            del __atran_cache[key]
        except KeyError:   # pragma: no cover
            # could happen in race conditions, working in parallel
            pass
        return

    modtime = str(os.path.getmtime(atranfile))
    if modtime not in __atran_cache.get(key, {}):
        return

    log.debug(f'Retrieving atmospheric transmission data from cache '
              f'({key[0]}, resolution {key[1]})')
    return __atran_cache.get(key, {}).get(modtime)


def store_atran_in_cache(atranfile, resolution, filename,
                         unsmoothed, wave, smoothed):
    """
    Store atmospheric transmission data in the atran cache.

    Parameters
    ----------
    atranfile : str
        File path to the atran file
    resolution : float
        Spectral resolution used for smoothing.
    filename : str
        Used to update ATRNFILE in FITS headers.
    unsmoothed : numpy.ndarray
        2D array containing the raw data from file
    wave : numpy.ndarray
        (nwave,) array of wavelengths.
    smoothed : numpy.ndarray
        (nwave,) array containing the smoothed atran data

    """
    global __atran_cache
    key = atranfile, int(resolution)
    log.debug(f'Storing atmospheric transmission data in cache '
              f'({key[0]}, resolution {key[1]})')
    __atran_cache[key] = {}
    modtime = str(os.path.getmtime(atranfile))
    __atran_cache[key][modtime] = (
        filename, unsmoothed, wave, smoothed)


def get_atran(header, resolution, filename=None,
              get_unsmoothed=False, atran_dir=None):
    """
    Retrieve reference atmospheric transmission data.

    PSG model files in the data/transmission directory should be named
    according to the altitude, ZA, and wavelengths for which they
    were generated, as:

        psg_[alt]K_[za]deg_[wmin]-[wmax]um.fits

    For example, the file generated for altitude of 41,000 feet,
    ZA of 45 degrees, and wavelengths between 5 and 28 microns
    should be named:

        psg_41K_45deg_5-28um.fits

    Model files are expected to be FITS images containing two
    rows. The first is wavenumber values in cm-1; the second is the
    fractional atmospheric transmission expected at that wavenumber.

    The procedure is:

        1. Identify model file by ZA and Altitude, unless override is
           provided.
        2. Read transmission data from file and smooth to expected
           spectral resolution.
        3. Return transmission array.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        ATRNFILE keyword is written to the provided FITS header,
        containing the name of the model file used.
    resolution : float
        Spectral resolution to which transmission data should be smoothed.
    filename : str, optional
        Atmospheric transmission file to be used.  If not provided,
        a default file will be retrieved from the data/transmission
        directory.  The file with the closest matching ZA and
        Altitude to the input data will be used.  If an override file
        is provided, it should be a FITS image file containing
        wavenumber and transmission data without smoothing.
    get_unsmoothed : bool, optional
        If True, return the unsmoothed atran data with original
        wavenumber array in addition to the smoothed array.
    atran_dir : str, optional
        Path to a directory containing model reference FITS files.
        If not provided, the default set of files packaged with the
        pipeline will be used.

    Returns
    -------
    atran : numpy.ndarray
        A (2, nw) array containing wavenumber and transmission data.
    unsmoothed : numpy.ndarray, optional
        An array containing wavenumber, unsmoothed
        transmission data, and any additional raw data rows,
        returned only if get_unsmoothed is set.
    """
    if filename is not None:
        if not goodfile(filename, verbose=True):
            log.warning(f'File {filename} not found; '
                        f'retrieving default')
            filename = None
    if filename is None:
        if not isinstance(header, fits.header.Header):
            log.error("Invalid header")
            return
        za_start = float(header.get('ZA_START', 0))
        za_end = float(header.get('ZA_END', 0))
        if za_start > 0 >= za_end:
            za = za_start
        elif za_end > 0 >= za_start:
            za = za_end
        else:
            za = 0.5 * (za_start + za_end)

        alt_start = float(header.get('ALTI_STA', 0))
        alt_end = float(header.get('ALTI_END', 0))
        if alt_start > 0 >= alt_end:
            alt = alt_start
        elif alt_end > 0 >= alt_start:
            alt = alt_end
        else:
            alt = 0.5 * (alt_start + alt_end)
        alt /= 1000

        log.info(f'Alt, ZA: {alt:.2f} {za:.2f}')
        true_value = [alt, za]

        if atran_dir is not None:
            if not os.path.isdir(str(atran_dir)):
                log.warning(f'Cannot find transmission directory: {atran_dir}')
                log.warning('Using default model set.')
                atran_dir = None
        if atran_dir is None:
            atran_dir = os.path.join(os.path.dirname(exes.__file__),
                                     'data', 'transmission')

        atran_files = glob.glob(os.path.join(atran_dir, 'psg*fits'))
        regex1 = re.compile(r'^psg_([0-9]+)K_([0-9]+)deg_5-28um\.fits$')

        # set up some values for tracking best match
        overall_val = np.inf
        best_file = None

        for f in atran_files:
            # check for filename match
            match = regex1.match(os.path.basename(f))
            if match is not None:
                match_val = 0
                for i in range(2):
                    # file alt or za
                    file_val = float(match.group(i + 1))
                    # check difference from true value
                    if true_value[i] != 0:
                        d_val = abs(file_val
                                    - true_value[i]) / true_value[i]
                    else:
                        d_val = abs(file_val - true_value[i])
                    match_val += d_val
                if match_val < overall_val:
                    overall_val = match_val
                    best_file = f

        log.info('Using nearest Alt/ZA')
        filename = best_file

    if filename is None:
        log.debug('No PSG file found')
        return

    # Read the model data from cache if possible
    log.info(f'Using PSG file: {filename}')
    atrandata = get_atran_from_cache(filename, resolution)
    if atrandata is not None:
        atranfile, unsmoothed, wave, smoothed = atrandata
    else:
        hdul = gethdul(filename, verbose=True)
        if hdul is None or hdul[0].data is None:
            log.error(f'Invalid data in model file {filename}')
            return
        unsmoothed = hdul[0].data
        hdul.close()

        atranfile = os.path.basename(filename)
        wave = unsmoothed[0]

        # convert wavenumber to wavelength
        awave = (unsmoothed[0] * units.kayser).to(
            units.um, equivalencies=units.spectral()).value

        # flip for monotonic increasing values
        awave = np.flip(awave)

        # smooth every row in the input model
        smoothed = unsmoothed.copy()
        for i in range(1, unsmoothed.shape[0]):

            trans = np.flip(unsmoothed[i])

            # smooth to constant resolution
            smooth_trans = smoothres(awave, trans, resolution)

            # flip back to match wavenumber
            smoothed[i] = np.flip(smooth_trans)

        store_atran_in_cache(filename, resolution, atranfile,
                             unsmoothed, wave, smoothed)

    hdinsert(header, 'ATRNFILE', atranfile)
    if not get_unsmoothed:
        return smoothed
    else:
        return smoothed, unsmoothed
