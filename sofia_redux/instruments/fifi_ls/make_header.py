# Licensed under a 3-clause BSD style license - see LICENSE.rst

from datetime import datetime
import os
import re

from astropy import log, units
from astropy.io import fits
from astropy.time import Time
import bottleneck as bn
import numpy as np
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities \
    import (robust_bool, valid_num, goodfile,
            natural_sort, date2seconds, hdinsert)

__all__ = ['create_requirements_table', 'clear_requirements_table',
           'get_keyword_comments', 'get_keyword_table',
           'get_keyword_comments_table', 'update_basehead',
           'order_headers', 'make_header']

__requirements_table = None
__requirements_file = None
__keyword_comments_file = None
__quick_comments = None


def create_requirements_table(default_file=None, comment_file=None,
                              reload=False):
    """
    Create the header keyword requirements definition table.

    Parameters
    ----------
    default_file : str, optional
        File path to the keyword definition file.  The default is
        fifi_ls/data/header_info/headerdef.dat
    comment_file : str, optional
        File path to the keyword comments file.  The default is
        fifi_ls/data/header_info/headercomment.dat.
    reload : bool, optional
        If set, data in cache will be ignored and files will be
        reloaded.
    """

    global __requirements_table
    global __requirements_file
    global __keyword_comments_file
    global __quick_comments

    create_table = (reload
                    or __requirements_table is None
                    or __requirements_file != default_file
                    or __keyword_comments_file != comment_file)

    if not create_table:
        return

    log.debug("Creating FIFI-LS keyword requirements table")
    clear_requirements_table()

    try:
        __requirements_table = get_keyword_table(filename=default_file).join(
            get_keyword_comments_table(filename=comment_file))
        __requirements_file = default_file
        __keyword_comments_file = comment_file
        __quick_comments = __requirements_table['comment'].to_dict()
        __requirements_table = __requirements_table.to_dict('index')
    except Exception as err:
        log.error(err)
        raise ValueError("Could not create requirements table")


def clear_requirements_table():
    """
    Clear all data from the requirements cache.
    """
    global __requirements_table
    global __requirements_file
    global __keyword_comments_file
    global __quick_comments

    __requirements_table = None
    __requirements_file = None
    __keyword_comments_file = None
    __quick_comments = None


def get_keyword_comments():
    """
    Get the keyword comments table from the cache.
    """
    global __quick_comments
    return __quick_comments


def get_keyword_table(filename=None):
    """
    Returns a dataframe containing the header requirements.

    Parameters
    ----------
    filename : str, optional
        File path to the keyword definition file.  The default is
        fifi_ls/data/header_info/headerdef.dat

    Returns
    -------
    pandas.DataFrame
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(fifi_ls.__file__),
                                'data', 'header_info', 'headerdef.dat')
    if not goodfile(filename, verbose=True, read=True):
        raise ValueError("invalid header definition file: %s" % filename)

    log.debug('Using keyword file: %s' % filename)

    columns = ['required', 'default', 'type', 'combine',
               'min', 'max', 'enum']
    types = {'int': int, 'integer': int, 'float': float, 'bool': robust_bool,
             'str': str, 'string': str, 'complex': complex}
    converters = {
        'required': robust_bool,
        'default': lambda x: None if x == '.' else x,
        'type': lambda x: types.get(x),
        'combine': lambda x: '' if x == '.' else x,
        'min': lambda x: float(x) if x != '.' else None,
        'max': lambda x: float(x) if x != '.' else None,
        'enum': lambda x: x.split('|') if x != '.' else []
    }
    table = pandas.read_csv(
        filename, delim_whitespace=True, comment='#',
        index_col=0, names=columns, converters=converters)
    table.index = table.index.str.upper().str.strip()
    table.enum = table.apply(
        lambda row: [row.type(x) for x in row.enum], axis=1)
    table.default = table.apply(lambda row: row.type(row.default), axis=1)
    table['key'] = table.index

    return table


def get_keyword_comments_table(filename=None):
    """
    Returns a dictionary containing header keyword comments.

    Parameters
    ----------
    filename : str, optional
        File path to the keyword comments file.  The default is
        fifi_ls/data/header_info/headercomment.dat.

    Returns
    -------
    pandas.DataFrame
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(fifi_ls.__file__),
                                'data', 'header_info', 'headercomment.dat')
    if not goodfile(filename, verbose=True, read=True):
        raise ValueError("invalid header comment file: %s" % filename)

    log.debug("Using keyword comment file %s" % filename)
    table = pandas.read_csv(
        filename, comment='#', index_col=0, names=['comment'],
        converters={'comment': str.strip}, skipinitialspace=True)
    table.index = table.index.str.upper().str.strip()
    return table


def clear_values(table):
    if table is None:
        return
    for row in table.values():
        row['value'] = None


def set_defaults(table):
    for row in table.values():
        if row.get('value') is None:
            row['value'] = row['default']


def get_keyword_values(basehead, headers,
                       default_file=None, comment_file=None):

    create_requirements_table(
        default_file=default_file,
        comment_file=comment_file,
        reload=False)

    global __requirements_table
    table = __requirements_table
    clear_values(table)

    if table is None:
        raise ValueError("Could not create requirements table")

    for key, row in table.items():
        table[key]['value'] = aggregate_key_value(basehead, headers, row)

    return table


def aggregate_key_value(basehead, headers, row):

    combine = row['combine']

    if combine == 'first' or len(headers) == 1:
        return value_from_header(basehead, row)

    elif combine == 'last':
        return value_from_header(headers[-1], row)

    elif combine == 'default':
        return row['default']

    elif combine == 'and':
        for header in headers:
            if not value_from_header(header, row):
                return False
        else:
            return True

    elif combine == 'or':
        for header in headers:
            if value_from_header(header, row):
                return True
        else:
            return False

    elif combine == 'concatenate':
        result = set()
        for header in headers:
            string_value = value_from_header(header, row)
            if not string_value:
                continue
            values = [x.upper().strip() for x in str(string_value).split(',')]
            for value in values:
                result.add(value)

        return ','.join(natural_sort(list(result)))

    elif combine == 'mean':
        result = []
        for header in headers:
            value = value_from_header(header, row)
            if value is not None:
                result.append(value)
        result = bn.nanmean(result)
        if not np.isfinite(result):
            result = row['default']
        else:
            result = row['type'](result)

        return result

    elif combine == 'sum':
        result = []
        for header in headers:
            value = value_from_header(header, row)
            if value is not None:
                result.append(value)
        result = bn.nansum(result)
        if not np.isfinite(result):
            result = row['default']
        else:
            result = row['type'](result)
        return result

    else:
        # return basehead value again
        return value_from_header(basehead, row)


def value_from_header(header, row, default=None):
    value = header.get(row['key'])
    if value is None:
        if default is not None:
            return default
        else:
            return None
    else:
        dtype = row['type']
        try:
            value = dtype(value)
        except (ValueError, TypeError, AttributeError):
            pass

    return value


def check_key(table, key):

    row = table.get(key)
    if row is None:
        log.warning("%s key is not in the keyword definitions" % key)
        # let it through
        return True

    value = row.get('value')

    if not row['required']:
        return True

    elif value is None:
        log.error('Required keyword %s not found' % key)
        return False

    dtype = row['type']
    if dtype is robust_bool:
        dtype = bool

    if not isinstance(value, dtype):
        log.error(
            "Required keyword %s has wrong type (value: %s). Should be %s" %
            (key, value, dtype))
        return False

    elif key == 'DATE-OBS':
        # special check for UTC 0 date (a common FIFI-LS glitch)
        try:
            mjd = Time(value).mjd
        except (ValueError, AttributeError, TypeError):
            mjd = 40587
        if int(mjd) == 40587:
            log.error("Required keyword DATE-OBS has wrong value (%s)"
                      % value)
            return False
        return True

    enum = row['enum']
    if len(enum) > 0:
        if value not in enum:
            log.error(
                "Required keyword %s has value (%s). Should be within [%s]"
                % (key, repr(value), ','.join(str(x) for x in enum)))
            return False

    if not np.isnan(row['min']) and value < row['min']:
        log.error("Required keyword %s has wrong value. Should be >= %s" %
                  (key, row['min']))
        return False

    if not np.isnan(row['max']) and value > row['max']:
        log.error("Required keyword %s has wrong value. Should be <= %s" %
                  (key, row['max']))
        return False

    # If we got here it's all good
    return True


def update_basehead(basehead, table, headers):
    """
    Update the base header with values that may be missing.

    Parameters
    ----------
    basehead : fits.Header
        FITS header to update
    headers : array_like of fits.Header
        List of headers from which to compile values
    table : dict
        Table of keywords and values

    Returns
    -------
    fits.Header
        Updated basehead
    """
    set_defaults(table)
    for key, row in table.items():
        hdinsert(basehead, key, row['value'], comment=row['comment'])

    comments = get_keyword_comments()

    # Add some FITS standard keys
    hdinsert(basehead, 'EQUINOX', 2000.0,
             comment='Equinox of celestial CS')
    hdinsert(basehead, 'RADESYS', 'FK5',
             comment='Celestial CS convention')
    hdinsert(basehead, 'TIMESYS', 'UTC',
             comment='Time system')
    hdinsert(basehead, 'TIMEUNIT', 's',
             comment='Time unit')
    hdinsert(basehead, 'XPOSURE', basehead.get('EXPTIME', 0),
             comment='Exposure time [s]')

    dateobs = basehead.get('DATE-OBS', 'UNKNOWN')
    utcstart = basehead.get('UTCSTART', '00:00:00')
    utcend = basehead.get('UTCEND', '00:00:00')
    datestr = str(dateobs).split('T')[0].strip()
    datebeg = '%sT%s' % (datestr, utcstart)
    dateend = '%sT%s' % (datestr, utcend)

    try:
        # Elapsed time in seconds
        telapse = (Time(dateend) - Time(datebeg)).to(units.s).value
    except ValueError:
        log.warning("Could not determine TELAPSE")
        telapse = 0.0

    hdinsert(basehead, 'DATE-BEG', datebeg)
    hdinsert(basehead, 'DATE-END', dateend)
    # format necessary for floating point annoyances
    hdinsert(basehead, 'TELAPSE', float("{:.5f}".format(telapse)))

    # copy aor to assc_aor and missn-id to assc_msn if single header
    if len(headers) == 1:
        aor = str(basehead.get('AOR_ID', 'UNKNOWN')).strip().upper()
        assc_aor = str(basehead.get('ASSC_AOR', 'UNKNOWN')).strip().upper()
        if assc_aor == 'UNKNOWN' and aor != 'UNKNOWN':
            hdinsert(basehead, 'ASSC_AOR', aor,
                     comment=comments['ASSC_AOR'])
        msn = str(basehead.get('MISSN-ID', 'UNKNOWN')).strip().upper()
        assc_msn = str(basehead.get('ASSC_MSN', 'UNKNOWN')).strip().upper()
        if assc_msn == 'UNKNOWN' and msn != 'UNKNOWN':
            hdinsert(basehead, 'ASSC_MSN', msn,
                     comment=comments['ASSC_MSN'])

    # add the current date/time
    utctime = Time(datetime.utcnow(), format='datetime').isot
    hdinsert(basehead, 'DATE', utctime.split('.')[0],
             comment=comments['DATE'])

    # set processing level to 2
    procstat = str(basehead.get('PROCSTAT', 'UNKNOWN')).strip().upper()
    if procstat not in ['LEVEL_3', 'LEVEL_4']:
        hdinsert(basehead, 'PROCSTAT', 'LEVEL_2',
                 comment='Processing status')

    # add raw file number
    filenum = str(basehead.get('FILENUM', 'UNKNOWN'))
    filename = str(basehead.get('FILENAME', 'UNKNOWN'))
    obsid = str(basehead.get('OBS_ID', 'UNKNOWN'))
    pattern = re.compile(r'[BR]([0-9]+)')
    match = pattern.findall(obsid)
    if filenum == 'UNKNOWN':
        if match and valid_num(match[-1]):
            filenum = match[-1].strip()
        else:
            if len(filename) >= 5:
                test = filename[:5].strip()
                if valid_num(test):
                    filenum = test
    if len(headers) > 1:
        filenums = [filenum]
        for h in headers:
            filenums.extend(str(h.get('FILENUM', 'UNKNOWN')).split('-'))
        filenums = natural_sort(list(np.unique(filenums)))
        filenums = [f for f in filenums if valid_num(f)]
        if len(filenums) > 1:
            filenum = filenums[0].strip() + '-' + filenums[-1].strip()
        elif len(filenums) == 1:
            filenum = filenums[0].strip()
        else:
            filenum = 'UNKNOWN'

    hdinsert(basehead, 'FILENUM', filenum,
             comment=comments['FILENUM'])

    # Modify the obsid
    obs = str(basehead.get('OBS_ID', 'UNKNOWN')).strip().upper()
    if not obs.startswith('P_'):
        assc_obs = str(basehead.get('ASSC_OBS', 'UNKNOWN')).strip().upper()
        if assc_obs == 'UNKNOWN' and obs != 'UNKNOWN':
            hdinsert(basehead, 'ASSC_OBS', obs,
                     comment=comments['ASSC_OBS'])
        hdinsert(basehead, 'OBS_ID', 'P_' + obs,
                 comment=comments['OBS_ID'])

    # Set the pipeline name and version
    hdinsert(basehead, 'PIPELINE', 'FIFI_LS_REDUX',
             comment=comments['PIPELINE'])
    hdinsert(basehead, 'PIPEVERS', fifi_ls.__version__.replace('.', '_'),
             comment=comments['PIPEVERS'])


def order_headers(headers):
    """
    Order headers based on contents.

    Return the earliest and the header list sorted by date

    Parameters
    ----------
    headers : array_like of fits.Header

    Returns
    -------
    2-tuple
       fits.Header : earliest header
       list of fits.Header : ordered headers
    """
    nhead = len(headers)
    if nhead == 1:
        return headers[0].copy(), [headers[0]]
    nodstyle = None
    dateobs, nodbeam = [], []
    for header in headers:

        if nodstyle is None:
            nodstyle = str(header.get('NODSTYLE'))
        dateobs.append(
            date2seconds(
                str(header.get('DATE-OBS', default='3000-01-01T00:00:00'))))
        nodbeam.append(str(header.get('NODBEAM', 'UNKNOWN')))

    # If C2NC2, get the earliest A header as the basehead
    # Otherwise, just use the earliest header
    index = np.argsort(dateobs)
    if nodstyle in ['C2NC2', 'ASYMMETRIC'] and 'A' in nodbeam:
        earliest_a = np.where(np.array(nodbeam)[index] == 'A')[0]
        earliest_a = 0 if earliest_a.size == 0 else earliest_a[0]
    else:
        earliest_a = 0

    earliest_idx = index[earliest_a]
    basehead = headers[earliest_idx].copy()

    # sort all headers by date-obs, including the basehead
    # This is used for C2NC2 mode, to get 'last' values, whether
    # in A or B nod
    sorted_headers = [headers[i] for i in index]

    return basehead, sorted_headers


def make_header(headers=None, checkheader=False, default_file=None,
                comment_file=None, check_all=False):
    """
    Standardize and combine input headers.

    Generates output headers for pipeline data products.

    The procedure is:

        1. Read the header keyword defaults and requirements from
           data/headerdef.dat and their associated default comments from
           data/headercomment.dat.
        2. Copy the earliest header for symmetric mode, or earliest A
           header for C2NC2 mode.  "Earliest" is defined by DATE-OBS.
        3. Loop through all keywords defined in headerdef.dat
            a. Get the value from the header as defined in the configuration
               algorithm.  The combination options are: and (for booleans),
               concatendate unique values with commas (for string values),
               use default value, use last value (i.e. latest according to
               DATE-OBS), or (for booleans), sum (for numerical values),
               mean (for numerical values), use the first value.  The default
               and most common case is to use the first value.  The
               combined keyword value is written to the output header.
            b. Check the value against requirements if desired.  Required
               keywords are checked for presence, checked against a
               specified data type (float, int, long, string, bool), and
               may additionally be checked against an enumerated value or
               min/max value range.  If requirements are not met, or if a
               defined keyword is not found in the input header, it is set
               to the default value in the output header.
        4. Some additional values are added or modified in the output
           header:

               - ASSC_AOR is copied from AOR_ID if not present
               - ASSC_MSN is copied from MISSN-ID if not present
               - DATE is set to the current date/time
               - PROCSTAT is set to LEVEL_2 (unless it is already LEVEL_3
                 or LEVEL_4
               - FILENUM is set from the raw filename if not present, or
                 from the range of input file numbers (first-last)
               - ASSC_OBS is copied from OBS_ID if not present
               - OBS_ID is prepended with P\\_ if not already done

    Parameters
    ----------
    headers : array_like of fits.Header, optional
        array of input FITS headers
    checkheader : bool, optional
        If True, will check keywords against SOFIA requirements.  If
        set, the return value will be a 2-tuple rather than a FITS
        header (see Return values).
    default_file : str, optional
        Path to the header keyword default file
    comment_file : str, optional
        Path to the header keyword comment file
    check_all : bool, optional
        If `checkheader` is True and a failure was encountered, keep
        checking the remainder of the keyword values and output warning
        messages.  Otherwise, the default is to return failure at the
        first bad keyword/value.

    Returns
    -------
    fits.Header
        a combined, standardized FITS header
        if checkheader is True then the return value will be a 2-tuple.
        The first element will be the header, and the second will be
        a boolean value indicating whether the header was created
        without any errors (False = errors were encountered).
    """
    if headers is None:
        headers = [fits.Header()]
    elif isinstance(headers, fits.Header):
        headers = [headers]
    elif not hasattr(headers, '__len__'):
        log.error("Invalid header")
        return (None, False) if checkheader else None
    elif len(headers) == 0:
        log.error("Empty list of headers")
        return (None, False) if checkheader else None

    for header in headers:
        if not isinstance(header, fits.Header):
            log.error("Invalid header in header list")
            return (None, False) if checkheader else None

    basehead, headers = order_headers(headers)
    table = get_keyword_values(basehead, headers,
                               default_file=default_file,
                               comment_file=comment_file)
    success = True
    if checkheader:
        for key in table.keys():
            if not check_key(table, key):
                success = False
                if not check_all:
                    break

    update_basehead(basehead, table, headers)

    return (basehead, success) if checkheader else basehead
