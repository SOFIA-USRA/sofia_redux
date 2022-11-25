# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import pandas

from sofia_redux.instruments import exes
from sofia_redux.toolkit.utilities import goodfile, hdinsert

__all__ = ['get_resolution']


def get_resolution(header):
    """
    Retrieve expected spectral resolving power.

    Requires spectral resolution files in exes/data/resolution for
    overriding cross-dispersed resolutions:

        - short_wave_resolution.txt
        - medium_wave_resolution.txt
        - long_wave_resolution.txt

    These files must have 2 columns: slit width (arcsec) and
    spectral resolving power (lambda / dlambda).  The file chosen is
    based on the central wavelength for the observation.

    The procedure is:

        1. Read resolution data from the configuration file matching
           the central wavelength of the observation (WAVECENT).
        2. Use SLITWID to find the closest resolution value.
        3. If no appropriate data is found, or the mode is not cross-dispersed,
           the value of the RP (if present) or RESOLUN keyword is returned.
           Otherwise, the best matching resolving power is returned.

    Parameters
    ----------
    header : fits.Header

    Returns
    -------
    float
        Expected resolving power.
    """
    if not isinstance(header, fits.Header):
        msg = 'Invalid header.'
        log.error(msg)
        raise ValueError(msg)

    required_keywords = ['WAVECENT', 'SLITWID', 'RESOLUN', 'INSTCFG']
    for key in required_keywords:
        if key not in header:
            msg = f'Header missing {key} keyword.'
            log.error(msg)
            raise ValueError(msg)

    instcfg = header['INSTCFG']
    slitwid = header['SLITWID']
    resolun = header['RESOLUN']
    wavecent = header['WAVECENT']

    # check for valid RP value: if present and reasonable,
    # use instead of resolun
    rp = header.get('RP', -9999)
    if rp > 500:
        resolun = rp

    if 'HIGH' not in str(instcfg).strip().upper():
        log.debug(f'Mode {instcfg}: returning header value.')
        return resolun

    if wavecent < 10.5:
        rfile = 'short_wave_resolution.txt'
    elif wavecent < 15.5:
        rfile = 'medium_wave_resolution.txt'
    else:
        rfile = 'long_wave_resolution.txt'

    resfile = os.path.join(os.path.dirname(exes.__file__), 'data',
                           'resolution', rfile)
    hdinsert(header, 'RESFILE', resfile)
    log.debug(f'Using resolution file: {resfile}')

    if not goodfile(resfile, verbose=True):
        msg = f'Cannot read resolution file: {resfile}'
        log.error(msg)
        raise ValueError(msg)

    # Find appropriate resolution
    log.debug(f"Loading resolution file: {resfile}")
    names = ['slitwid', 'res']
    df = pandas.read_csv(resfile, comment='#', names=names,
                         delim_whitespace=True)
    if df.empty:
        log.debug('No resolution data; returning header value.')
        return resolun

    resolution = float(df.loc[
        (df.slitwid - slitwid).abs().idxmin()]['res'])

    log.debug(f'Found resolution: {resolution}')
    return resolution
