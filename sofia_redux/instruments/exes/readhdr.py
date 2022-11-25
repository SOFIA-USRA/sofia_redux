# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import re

from astropy import log
from astropy.io.fits.header import Header
from astropy.time import Time
from astropy.utils.data import download_file
import numpy as np
import pandas as pd

from sofia_redux.instruments import exes
from sofia_redux.instruments.exes.utils import \
    set_elapsed_time, parse_central_wavenumber
from sofia_redux.toolkit.utilities.fits import hdinsert
from sofia_redux.toolkit.utilities.func import goodfile, robust_bool

__all__ = ['readhdr']

# back up download URL for non-source installs
DATA_URL = 'https://sofia-exes-reference.s3-us-gov-west-1.amazonaws.com/'


def readhdr(header, check_header=True,
            config_file=None, comments_file=None):
    r"""
    Read and update an EXES FITS header.

    Keywords that must be present in the output header are defined in
    exes/data/header/headerdef.dat.  Default values and acceptable
    ranges are also defined there.  For each keyword defined there,
    the header is checked for a value.  If it is found, the value is
    added to the output header.

    If checkreq=True, it is checked against the allowed values.  If
    it is out of range, header_ok will be set to False.  If the
    value is not found, the default value is added.  If it is missing
    and required, header_ok will be set to False, but the value
    will be updated.

    Some additional special values are added to the output header:

        * Comments associated with all keywords are read from
          exes/data/header/headercomment.dat.
        * The full path to the package containing this routine is
          stored under the key 'PKGPATH'.
        * The full path to the package/data directory is stored
          under the key 'DATAPATH'.
        * Distortion correction parameters are read from
          exes/data/tort/\*.dat and added to the output header.
          Some additional parameters are calculated from these and
          other header parameters and added to the output header.
        * The slit angle is converted from degrees to cm using
          the values in exes/data/slitval.dat.
        * Filenames for BPM, LINFILE, and DARKFILE are looked up
          by date from exes/data/caldefault.dat.
        * The current date is added under the key 'DATE'.

    Parameters
    ----------
    header : fits.Header
        The FITS header to check and update.
    check_header : bool, optional
        If True, check against header requirements and return success or
        failure status.
    config_file : str, optional
        Path to the headerdef.dat header defaults configuration file
    comments_file : str, optional
        Path to the headercomment.dat header keyword comments file

    Returns
    -------
    updated_header : fits.Header
        The updated FITS header
    success : bool, optional
        True if header checks succeeded; False if they did not.

    Raises
    ------
    ValueError if header is incorrectly formatted.
    """
    if not isinstance(header, Header):
        raise ValueError(f"Header is not {Header}")

    header_config = _get_header_configuration(
        config_file=config_file, comments_file=comments_file)

    new_header = header.copy()
    success = True
    for keyword in header_config.index:
        value = header.get(keyword)
        default_value = _get_default(keyword, header_config)
        comment = header_config.loc[keyword].comment

        # try to convert to expected type
        if value is not None:
            try:
                value = header_config.loc[keyword].type(value)
            except (ValueError, TypeError):
                # leave it if it fails
                pass

        if check_header:
            success &= _checkreq(keyword, value, header_config)

        if value is not None:
            hdinsert(new_header, keyword, value, comment=comment)
        elif default_value is not None:
            hdinsert(new_header, keyword, default_value, comment=comment)

    _standardize_values(new_header)
    _set_decimal_date(new_header)
    _process_instrument_configuration(new_header)
    _process_slit_configuration(new_header)
    _process_tort_configuration(new_header)
    _add_configuration_files(new_header)

    if check_header:
        return new_header, success
    else:
        return new_header


def _get_configuration_file(filename, subdir=None, check=True):
    """Get a configuration file from exes path."""
    datadir = os.path.join(os.path.dirname(exes.__file__), 'data')
    if subdir is not None:
        datadir = os.path.join(datadir, subdir)

    if check and not os.path.isdir(datadir):
        raise ValueError(f"{datadir} does not exist")

    datafile = os.path.join(datadir, filename)
    if check and not goodfile(datafile):
        raise ValueError(f"Could not read: {datafile}")
    return datafile


def _get_header_configuration(config_file=None, comments_file=None):
    """Get header configuration from a stored definition."""
    if config_file is None:
        config_file = _get_configuration_file('headerdef.dat', subdir='header')
    else:
        if not goodfile(config_file, verbose=True):
            raise ValueError(
                f"Header default file does not exist: {config_file}")

    if comments_file is None:
        comments_file = _get_configuration_file(
            'headercomment.dat', subdir='header')
    else:
        if not goodfile(comments_file, verbose=True):
            raise ValueError(f'header comments file does not exist: '
                             f'{comments_file}')

    columns = ['required', 'default', 'type', 'min', 'max', 'enum']
    types = {'int': int, 'integer': int, 'float': float, 'bool': robust_bool,
             'str': str, 'string': str, 'complex': complex, 'double': float}
    converters = {'required': robust_bool,
                  'default': lambda x: None if x == '.' else x,
                  'type': lambda x: types.get(x),
                  'min': lambda x: float(x) if x != '.' else None,
                  'max': lambda x: float(x) if x != '.' else None,
                  'enum': lambda x: x.split('|') if x != '.' else []}
    df = pd.read_csv(config_file, delim_whitespace=True, comment='#',
                     index_col=0, names=columns, converters=converters)
    df.index = df.index.str.upper().str.strip()

    converters = {'comment': str.strip, 'keyword': str.strip}
    comments = pd.read_csv(comments_file, comment='#',
                           names=['keyword', 'comment'],
                           converters=converters,
                           sep='^([^,]+),',
                           engine='python')
    comments.set_index('keyword', inplace=True)
    comments.index = comments.index.str.upper()

    return df.join(comments)


def _get_default(keyword, table):
    """
    Get a keyword default value from the keyword defaults table.

    Parameters
    ----------
    keyword : str
    table : pandas.DataFrame

    Returns
    -------
    value : str, float, int, or bool
        The default value.
    """
    if keyword not in table.index:
        log.error(f"Keyword {keyword} not found in keywords default table")
        return
    dtype = table.loc[keyword]['type']
    if dtype is bool:
        dtype = robust_bool
    value = table.loc[keyword]['default']

    # handle missing default, for keywords that should be
    # tracked but not added if missing
    if value is None:
        return None

    try:
        value = dtype(value)
    except (ValueError, TypeError, AttributeError):
        value = None

    # handle poorly specified default
    if value is None:
        if dtype is robust_bool:  # pragma: no cover
            # shouldn't be reachable, for robust bool type
            value = False
        elif dtype in [float, int]:
            value = dtype(-9999)
        elif dtype is str:  # pragma: no cover
            # shouldn't be reachable, for str type
            value = 'UNKNOWN'
    return value


def _checkreq(key, value, table):
    """
    Check whether the value of a key fits requirements.

    Parameters
    ----------
    key : str
        Key name
    value : str, float, int, or bool
        Value to test
    table : pandas.DataFrame
        Keyword defaults table

    Returns
    -------
    bool
        True if value fits the requirements, False otherwise.  If the
        keyword is not found in the keyword defaults table or is not
        required, then None will be returned.
    """
    # special check for UTC 0 date (a common FIFI-LS glitch)
    if key == 'DATE-OBS':
        try:
            mjd = Time(value).mjd
        except (ValueError, AttributeError, TypeError):
            mjd = 40587
        if int(mjd) == 40587:
            log.error(f"Required keyword DATE-OBS has wrong value ({value})")
            return False

    if key not in table.index:  # If there are no rules, allow it to go through
        log.warning(f"{key} keyword not found in keyword defaults table")
        return True
    row = table.loc[key]

    if not row['required']:
        return True

    if value is None:
        log.error(f'Required keyword {key} has missing value')
        return False

    rtype = row['type']
    rtype = bool if rtype is robust_bool else rtype
    if not isinstance(value, rtype):
        # allow ints for float values
        if not (rtype is float and isinstance(value, int)):
            log.error(
                f"Required keyword {key} has wrong type (value: {value}). "
                f"Should be {row['type']}")
            return False

    # Check enum
    if len(row['enum']) > 0:
        dtype = row['type']
        dtype = robust_bool if dtype is bool else dtype
        check_enum = [dtype(x) for x in row['enum']]
        if value not in check_enum:
            log.error(f"Required keyword {key} has wrong value {value}. "
                      f"Should be within [{','.join(row['enum'])}]")
            return False

    # Check min
    if not np.isnan(row['min']) and value < row['min']:
        log.error(f"Required keyword {key} has wrong value {value}. "
                  f"Should be >= {row['min']}")
        return False

    # Check max
    if not np.isnan(row['max']) and value > row['max']:
        log.error(f"Required keyword {key} has wrong value {value}. "
                  f"Should be <= {row['min']}")
        return False
    return True


def _set_decimal_date(header):
    """
    Set the FDATE key in the header to a decimal value representing date.

    Parameters
    ----------
    header : fits.Header
        Header to update.
    """

    # Decimal date
    date = header.get('DATE-OBS', 'UNKNOWN')
    fdate = header['FDATE']
    try:
        time = Time(date, scale='utc', format='isot').isot
    except ValueError:
        log.warning(f'DATE-OBS {date} not understood, using date={fdate}')
        return

    pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})')
    regex = pattern.match(time)
    if regex is None:  # pragma: no cover
        # shouldn't be reachable, if Time was created correctly
        log.warning(f'DATE-OBS {date} not understood, using date={fdate}')
        return

    y, m, d, h = map(int, regex.groups())
    if y > 100:
        y -= 2000

    fdate = round(y + (m / 1e2) + (d / 1e4) + (h / 1e6), 6)
    header['FDATE'] = fdate


def _standardize_values(header):
    """
    Reformat certain values.

    Header is updated in-place.

    Parameters
    ----------
    header : fits.Header
        Header to update.
    """
    package_path = os.path.dirname(exes.__file__)
    header['PKGPATH'] = package_path
    header['DATAPATH'] = os.path.join(package_path, 'data')

    header['ASSC_AOR'] = str(header.get('AOR_ID', 'UNKNOWN'))
    header['CARDMODE'] = str(header.get('CARDMODE', 'UNKNOWN')).upper().strip()
    header['INSTMODE'] = str(header.get('INSTMODE', 'UNKNOWN')).upper().strip()

    float_keys = ['EXPTIME', 'WAVENO0', 'XDDGR', 'GRATANGL',
                  'XDDELTA', 'SDEG', 'HRFL0', 'XDFL0',
                  'EFL0', 'PIXELWD', 'SLITVAL', 'SLITWID',
                  'HRG', 'WNO0', 'HRDGR', 'HRR', 'FDATE']
    for key in float_keys:
        if key in header:
            try:
                header[key] = float(header[key])
            except Exception as err:
                header[key] = -9999.0
                log.warning(f"Unable to convert {key} key to float: {err}")
        else:
            header[key] = -9999.0

    instcfg = header['INSTCFG'].upper().strip()
    if instcfg in ['HI-MED', 'HIGH-MED', 'HIMED']:
        header['INSTCFG'] = 'HIGH_MED'
    elif instcfg in ['HI-LO', 'HIGH-LOW', 'HILO']:
        header['INSTCFG'] = 'HIGH_LOW'
    elif instcfg in ['MED']:
        header['INSTCFG'] = 'MEDIUM'
    elif instcfg in ['LO']:
        header['INSTCFG'] = 'LOW'
    elif instcfg in ['CAM', 'PUP']:
        header['INSTCFG'] = 'CAMERA'
    else:
        header['INSTCFG'] = instcfg

    nexp = int(header.get('NEXP', -9999))
    header['NEXP'] = 1 if nexp == -9999 else nexp

    # For SOFIA
    header['EFL0'] = 3000.0

    # Check for pinhole mode
    header['PINHOLE'] = False  # not used for EXES yet

    # Add the current date/time
    header['DATE'] = Time.now().isot[:-4]

    # Set the elapsed time within the file
    set_elapsed_time(header)

    # If raw filename has not yet been added, add it now
    if 'RAWFNAME' not in header:
        header['RAWFNAME'] = (str(header.get('FILENAME', 'UNKNOWN')),
                              'Raw file name')


def _process_instrument_configuration(header):
    """
    Update header based on instrument configuration.

    Header is updated in-place.

    Parameters
    ----------
    header : fits.Header
    """
    instcfg = header['INSTCFG']
    if instcfg in ['MEDIUM', 'HIGH_MED', 'LOW', 'HIGH_LOW']:
        header['GRATANGL'] = header['ECHELLE']
        if 'LOW' in instcfg:
            header['XDDGR'] = header['XDLRDGR']  # LOW
        else:
            header['XDDGR'] = header['XDMRDGR']  # MEDIUM

    if instcfg != 'CAMERA':
        w0 = parse_central_wavenumber(header)
        xddgr = float(header['XDDGR'])
        gangle = float(header['GRATANGL'])
        iorder = int(np.round(2 * xddgr * np.sin(np.radians(gangle)) * w0))

        if instcfg in ['MEDIUM', 'HIGH_MED']:
            sinang = iorder / (2 * xddgr * w0)
            theta = np.arcsin(sinang)
        else:
            theta = np.radians(gangle)

        header['XDR'] = np.tan(theta + float(header['XDDELTA']))

        if instcfg == 'HIGH_LOW':
            # XDR seems underestimated by 2 for high-low mode
            header['XDR'] /= 2

        # assign a rough resolution if it is missing entirely
        # or has a bad value
        res = header.get('RESOLUN', -9999)
        if res < 500:
            log.warning(f'RESOLUN has incorrect value: {res}')
            if 'HIGH' in instcfg:
                header['RESOLUN'] = 75000
            elif instcfg == 'LOW':
                header['RESOLUN'] = 2000
            else:
                header['RESOLUN'] = 10000
            log.warning(f"Assigning default value: {header['RESOLUN']}")


def _process_slit_configuration(header):
    """
    Update the SLITVAL keyword based on configuration and header.

    Header is updated in-place.

    Parameters
    ----------
    header : fits.Header
    """
    slit_file = os.path.join(
        os.path.dirname(exes.__file__), 'data', 'slit', 'slitval.dat')
    if not goodfile(slit_file, verbose=True):
        raise ValueError(f"Could not read slit configuration "
                         f"file: {slit_file}")

    columns = ['date', 'limit', 'div1', 'div2', 'c1', 'c2']
    columns.extend([f'v{i}' for i in range(24)])

    df = pd.read_csv(slit_file, delim_whitespace=True, comment='#',
                     names=columns, dtype=float)
    row = df[df['date'] > header['fdate']].iloc[0]

    slit = float(header['SDEG'])
    if slit < row['limit']:
        c = row['c1']
        m = row['div1']
    else:
        c = row['c2']
        m = row['div2']

    islit = int(slit / m) + int(c) - 1
    slitval = 0
    if 0 <= islit < 24:
        slitval = row[f'v{islit}']

    if slitval == 0:
        log.warning("Slit angle out of range")
        slitval = 0.01

    header['SLITVAL'] = slitval


def _process_tort_configuration(header):
    """
    Set tort parameters from central wave number and date.

    Header is updated in-place.

    Parameters
    ----------
    header : fits.Header
    """
    tort_directory = os.path.join(
        os.path.dirname(exes.__file__), 'data', 'tort')
    tort_name = header['INSTCFG'].replace('_', '').lower()
    tort_file = os.path.join(tort_directory, 'tortparm_' + tort_name + '.dat')
    if not goodfile(tort_file, verbose=True):
        log.warning(f"Cannot read tort file: {tort_file}")
        log.warning('Using low mode parameters.')
        tort_file = os.path.join(tort_directory, 'tortparm_low.dat')

    columns = ['date', 'hrfl0', 'xdfl0', 'slitrot',
               'krot', 'brl', 'x0brl', 'y0brl', 'hrr', 'detrot']
    df = pd.read_csv(tort_file, delim_whitespace=True, comment='#',
                     names=columns, dtype=float)
    row = df[df['date'] > header['fdate']].iloc[0]
    log.info(f"Using tort parameters from: {tort_file} {row['date']}")

    header['BRL'] = row['brl']
    header['X0BRL'] = row['x0brl']
    header['Y0BRL'] = row['y0brl']

    # Only use default values if not already set for the following keys
    override_keys = ['KROT', 'SLITROT', 'HRFL0', 'XDFL0', 'HRR', 'DETROT']
    for key in override_keys:
        if np.isclose(header[key], -9999):
            header[key] = row[key.lower()]

    hrfl0 = float(header['HRFL0'])
    xdfl0 = float(header['XDFL0'])
    efl0 = float(header['EFL0'])
    slitval = float(header['SLITVAL'])
    hrg = float(header['HRG'])

    # This value is a holdover from the TEXES pipeline,
    # which used a focal reducer. EXES doesn't have one,
    # but the calculation is preserved here for historical
    # reasons.
    fred = 1.0
    header['HRFL'] = hrfl0 / fred
    header['XDFL'] = xdfl0 / fred
    header['EFL'] = efl0 / fred
    header['HRDGR'] = 0.3 * 2.54 * 0.996 * np.cos(hrg)

    # preferentially use slit width from estimated header value
    # instead of recalculating:
    slitwid = header.get('SLIT_AS', -9999)
    if slitwid == -9999:
        slitwid = slitval / (efl0 * 4.848e-06)
    header['SLITWID'] = slitwid

    # set platescale value by mode
    instcfg = header['INSTCFG']
    pltscale = 0.201
    if 'HIGH' in instcfg:
        # input angle
        alpha = np.radians(header['ECHELLE'])

        # difference between input and output beams: 5.43 degrees
        delta = np.radians(5.43)

        # output angle
        beta = alpha + delta

        # corrected plate scale
        header['PLTSCALE'] = pltscale * np.cos(beta) / np.cos(alpha)
    else:
        header['PLTSCALE'] = pltscale

    # set solid angle per pixel from plate scale and slit width
    header['OMEGAP'] = header['PLTSCALE'] * slitwid * (4.848e-06) ** 2


def _add_configuration_files(header):
    """
    Add file paths to configuration files.

    Header is updated in-place.

    Parameters
    ----------
    header : fits.Header
    """
    datapath = os.path.join(os.path.dirname(exes.__file__), 'data')
    default_file = os.path.join(datapath, 'caldefault.dat')
    if not goodfile(default_file, verbose=True):
        raise ValueError(f"Could not read default file: {default_file}")

    columns = ['date', 'bpmfile', 'darkfile', 'linfile']
    df = pd.read_csv(default_file, delim_whitespace=True,
                     comment='#', names=columns)
    df['date'] = df['date'].apply(float)
    row = df[df['date'] > header['fdate']].iloc[0]

    bpmfile = os.path.join(datapath, 'bpm', row['bpmfile'])
    if goodfile(bpmfile, verbose=True):
        # standard cal path for source distribution
        header['BPM'] = bpmfile
    else:
        # retrieve remotely if needed
        header['BPM'] = _download_cache_file(row['bpmfile'])

    linfile = os.path.join(datapath, 'lincoeff', row['linfile'])
    if goodfile(linfile, verbose=True):
        header['LINFILE'] = linfile
    else:
        header['LINFILE'] = _download_cache_file(row['linfile'])

    darkfile = os.path.join(datapath, 'dark', row['darkfile'])
    if goodfile(darkfile, verbose=True):
        header['DRKFILE'] = darkfile
    else:
        header['DRKFILE'] = _download_cache_file(row['darkfile'])


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
