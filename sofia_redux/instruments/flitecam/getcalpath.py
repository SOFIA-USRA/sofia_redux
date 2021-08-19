# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections import OrderedDict
import os

from astropy import log
from astropy.utils.data import download_file
import pandas

from sofia_redux.instruments import flitecam as fdrp

__all__ = ['getcalpath']

# back up download URL for non-source installs
DATA_URL = 'https://sofia-flitecam-reference.s3-us-gov-west-1.amazonaws.com/'


def getcalpath(header):
    """
    Return the path of the ancillary files used for the pipeline.

    Looks up default calibration files and other reference data in
    the calibration data path, based on the characteristics of the
    observation, as recorded in the input header.

    The output format is designed to be compatible with FORCAST
    configurations.

    Parameters
    ----------
    header : astropy.io.fits.Header

    Returns
    -------
    collections.OrderedDict
        name : str
            Name of the config mode.
        gmode : int
            The grism mode to be used during the reduction.
        cnmode : int
            The chop/nod mode to be used during the reduction.
        dateobs : int
            The observation date, in YYYYMMDD format.
        obstype : str
            Observation type.
        srctype : str
            Source type.
        spectel : str
            Name of the spectral element used for the observation.
        slit : str
            Name of the slit used for the observation.
        pathcal : str
            Base file path for calibration data.
        kwfile : str
            File path to the keyword definition file.
        linfile : str
            File path to the linearity correction file.
        maskfile : str
            File path to the grism order mask file.
        wavefile : str
            File path to the grism wavelength calibration file.
        resolution : float
            Spectral resolution for the grism mode
        respfile : str
            File path to the grism instrument response file.
        slitfile : float
            File path to the grism slit function file.
        linefile : str
            File path to a line list for wavelength calibration.
        wmin : int
            Approximate minimum wavelength for the instrument. Used
            to identify matching ATRAN files.
        wmax : int
            Approximate maximum wavelength for the instrument. Used
            to identify matching ATRAN files.
        error : bool
            True if there was an error retrieving the files.
    """
    path = os.path.join(os.path.dirname(fdrp.__file__), 'data')

    result = OrderedDict(
        (('name', ''), ('gmode', -1), ('cnmode', ''), ('dateobs', 99999999),
         ('obstype', ''), ('srctype', ''), ('spectel', ''), ('slit', ''),
         ('pathcal', ''), ('kwfile', ''), ('linfile', ''),
         ('maskfile', ''), ('wavefile', ''),
         ('respfile', ''), ('slitfile', ''), ('linefile', ''),
         ('waveshift', 0.), ('resolution', 0.),
         ('wmin', 1), ('wmax', 6),
         ('error', False)))

    result['spectel'] = header.get('SPECTEL1', '').upper().strip()
    result['slit'] = header.get('SPECTEL2', '').upper().strip()

    date = header.get('DATE-OBS', '').replace('T', ' ').replace('-', ' ')
    date = date.split()
    dateobs = 99999999
    if len(date) >= 3:
        try:
            dateobs = int(''.join(date[0:3]))
        except ValueError:
            pass
    result['dateobs'] = dateobs

    instcfg = header.get('INSTCFG', 'UNKNOWN').strip().upper()
    if instcfg in ['SPECTROSCOPY', 'GRISM']:
        name = 'GRI'
        gmode = 1
    else:
        name = 'IMA'
        gmode = -1
    result['name'] = name
    result['gmode'] = gmode

    # also read and store the sky mode, obstype, and srctype
    # from the header
    result['cnmode'] = header.get('INSTMODE', 'UNKNOWN').strip().upper()
    result['obstype'] = header.get('OBSTYPE', 'OBJECT').strip().upper()
    result['srctype'] = header.get('SRCTYPE', 'UNKNOWN').strip().upper()

    # read the top-level default file
    result['pathcal'] = os.path.join(path, '')
    calfile_default = os.path.join(path, 'caldefault.txt')
    if not os.path.isfile(calfile_default):
        msg = f"Problem reading default file " \
              f"{calfile_default}"
        log.warning(msg)
        result['error'] = True
        return result

    calcols = ['name', 'kwfile', 'linfile']
    df = pandas.read_csv(calfile_default, delim_whitespace=True,
                         comment='#', names=calcols, index_col=0)
    table = df[(df.index >= dateobs) & (df['name'] == result['name'])]
    if len(table) == 0:
        msg = f"Problem reading defaults for " \
              f"{result['name']}on date {dateobs} from {calfile_default}"
        log.warning(msg)
        result['error'] = True
        return result

    # take the first applicable date
    row = table[table.index == table.index.min()].sort_index().iloc[0]
    for f in calcols:
        if f.endswith('file') and f in row and row[f] != '.':
            # for source distributions, the file should be in
            # the standard calpath
            expected = os.path.join(path, row[f])
            if os.path.isfile(expected):
                result[f] = expected
            else:
                # for public pip/conda distributions, it may need
                # to be downloaded from S3
                result[f] = _download_cache_file(row[f])

    # Read additional grism defaults into result
    if result['gmode'] > 0:
        _get_grism_cal(path, result)

    return result


def _get_grism_cal(pathcal, result):
    """
    Return the path of the ancillary files used for the grism pipeline.

    Looks up additional default calibration files and other reference data
    needed for grism observations.  The input parameters are assumed to be
    passed from the getcalpath function: they should already be formatted
    as necessary.

    Parameters
    ----------
    header : astropy.io.fits.Header
    pathcal : str
        Absolute file path to calibration data directory that contains
        the default calibration files.
    result : collections.OrderedDict
        The configuration structure to update with grism values.
    """
    spectel = result['spectel']
    slit = result['slit']
    dateobs = result['dateobs']

    pathcal = os.path.join(pathcal, 'grism')
    calfile_default = os.path.join(pathcal, 'caldefault.txt')
    if not os.path.isfile(calfile_default):
        msg = f"Problem reading default file {calfile_default}."
        log.warning(msg)
        result['error'] = True
        return

    calcols = ['spectel', 'slit', 'maskfile', 'wavefile',
               'respfile', 'linefile', 'waveshift', 'resolution']
    df = pandas.read_csv(calfile_default, delim_whitespace=True,
                         comment='#', names=calcols, index_col=0)
    table = df[(df.index >= dateobs)
               & (df['spectel'] == spectel)
               & (df['slit'] == slit)]
    if len(table) == 0:
        msg = 'getcalpath - Problem reading defaults ' \
              'for %s, %s ' % (spectel, slit)
        msg += 'on date %s from %s' % (dateobs, calfile_default)
        log.warning(msg + '\nRoutine failed. Returning default dict')
        result['error'] = True
        return

    # take the first applicable date
    row = table[table.index == table.index.min()].sort_index().iloc[0]
    for f in calcols:
        if f.endswith('file') and f in row and row[f] != '.':
            # for source distributions, the file should be in
            # the standard calpath
            expected = os.path.join(pathcal, row[f])
            if os.path.isfile(expected):
                result[f] = expected
            else:
                # for public pip/conda distributions, it may need
                # to be downloaded from S3
                result[f] = _download_cache_file(row[f])
        elif f in ['waveshift', 'resolution'] and f in row and row[f] != '.':
            try:
                result[f] = float(row[f])
            except ValueError:
                msg = f'Problem reading {f} for {spectel}, {slit} ' \
                      f'on date {dateobs} from {calfile_default}'
                log.warning(msg)
                result['error'] = True


def _download_cache_file(filename):
    basename = os.path.basename(filename)
    url = f'{DATA_URL}{basename}'

    try:
        cache_file = download_file(url, cache=True, pkgname='sofia_redux')
    except (OSError, KeyError):
        # return basename only if file can't be downloaded;
        # pipeline will issue clearer errors later
        cache_file = basename
        log.warning(f'File {cache_file} could not be downloaded from {url}')

    return cache_file
