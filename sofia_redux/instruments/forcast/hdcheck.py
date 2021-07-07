# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections.abc import Iterable
import os

from astropy import log
from astropy.io import fits
from numpy import isfinite
from pandas import Series, DataFrame

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.hdrequirements import hdrequirements

__all__ = ['validate_condition', 'validate_compound_condition',
           'validate_keyrow', 'validate_header',
           'validate_file', 'hdcheck']


def validate_condition(header, condition, dripconf=False):
    """
    Return if a keyword header value meets a condition

    Retrieves the keyword named in the condition from the header.
    The value from the condition is converted to the same type
    as that in the header before a comparison is performed using
    the comparison operator from the condition in the following
    order:

        <header value> <condition operator> <condition value>

    The result of the equality is returned

    Parameters
    ----------
    header : astropy.io.header.Header
    condition : 3-tuple
        of the form (KEYWORD, comparison operator, value)
    dripconf : bool, optional
        Will check the configuration file for the keyword in
        addition to the header

    Returns
    -------
    bool
        True if the condition is met, False otherwise
    """
    if len(condition) == 0:
        return True
    if not isinstance(header, fits.header.Header):
        log.error("Invalid FITS header")
        return False
    if not isinstance(condition, tuple) or len(condition) != 3:
        log.error("Invalid condition recieved: %s" % repr(condition))
        return False
    keyword, equality, requirement = condition
    if equality not in ['<', '<=', '>', '>=', '!=', '==']:
        log.error("Invalid comparison operator in condition: %s"
                  % repr(condition))
        return False
    value = getpar(header, keyword) if dripconf else header.get(keyword)
    if value is None:
        return False

    value_type = type(value)
    try:
        if value_type == bool:
            if requirement.strip() == '0':
                requirement = False
        requirement = value_type(requirement)
    except (TypeError, ValueError):
        log.error("Could not convert %s to %s" %
                  (requirement, value_type))
        return False

    operation = {
        '<': lambda x, y: x < y,
        '<=': lambda x, y: x <= y,
        '>': lambda x, y: x > y,
        '>=': lambda x, y: x >= y,
        '==': lambda x, y: x == y,
        '!=': lambda x, y: x != y}

    # any types that survived the casting will be comparable
    if not operation[equality](value, requirement):
        return False
    return True


def validate_compound_condition(header, conditions, dripconf=False):
    """
    Checks the AND/OR conditions of keyword definitions is met by header

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    conditions : list of list of tuple of str
    dripconf : bool, optional
        Will check the configuration file for keywords in
        addition to the header

    Returns
    -------
    bool
        True if the condition is met, False otherwise
    """
    if len(conditions) == 0:
        return True
    for or_condition in conditions:
        for and_condition in or_condition:
            ok = validate_condition(
                header, and_condition, dripconf=dripconf)
            if not ok:
                break
        else:
            return True
    else:
        return False


def validate_keyrow(header, keyrow, dripconf=False):
    """
    Check if a header of a FITS file matches a keyword requirement

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    keyrow : pandas.Series
        Single key series from keywords table
    dripconf : bool
        If True, use the drip configuration to check for a keyword
        in addition to the header
    Returns
    -------
    bool
        True if header follows requirement, False if not
    """
    if not isinstance(header, fits.header.Header):
        log.error("Header - not an astropy Header: %s" % type(header))
        return False
    elif not isinstance(keyrow, Series):
        log.error("Keywords - not a pandas Series: %s" % type(keyrow))
        return False
    required_cols = {'condition', 'enum', 'format',
                     'max', 'min', 'required', 'type'}
    missing = required_cols.difference(set(keyrow.index))
    if missing:
        log.error("Columns missing in the keyword table: %s" %
                  ', '.join(missing))
        return False

    if not keyrow['required']:
        return True

    # Check other keyword requirements
    required = validate_compound_condition(
        header, keyrow['condition'], dripconf=dripconf)
    if not required:
        return True

    value = getpar(header, keyrow.name) if getpar else header.get(keyrow.name)

    # check presence
    if value is None:
        msg = "Parameter %s is missing" % (keyrow.name)
        log.error(msg)
        return False

    # check type
    if type(value) != keyrow['type']:
        if type(value) not in [float, int] or \
                keyrow['type'] not in [float, int]:
            msg = "Parameter %s %s has wrong type %s" % (
                keyrow.name, keyrow['type'], type(value))
            if not dripconf:
                log.error(msg)
            else:
                log.error(msg + ' for use with dripconf option %s' % str)
            return False

    # check enum
    if len(keyrow['enum']):
        checkvals = [str(val).upper() for val in keyrow['enum']]
        strval = str(value).upper()
        if strval not in checkvals:
            log.error("Parameter %s=%s is not within {%s}" %
                      (keyrow.name, repr(value), ','.join(keyrow['enum'])))
            return False

    # check min/max
    if keyrow['type'] in [int, float, bool]:
        if isfinite(keyrow['min']):
            if value < keyrow['min']:
                log.error("Parameter %s=%s is lower than %s" %
                          (keyrow.name, value, keyrow['min']))
                return False

        if isfinite(keyrow['max']):
            if value > keyrow['max']:
                log.error("Parameter %s=%s is higher than %s" %
                          (keyrow.name, value, keyrow['max']))
                return False
    return True


def validate_header(header, keywords, dripconf=False):
    """
    Validate all keywords in a header against the keywords table

    Parameters
    ----------
    header : astropy.io.fits.header.Header
    keywords : pandas.DataFrame
    dripconf : bool
        If True, will check the configuration file for the required
        parameter before checking headers (using getpar)

    Returns
    -------
    bool
        True if all keywords in the header were validated, False otherwise
    """
    if not isinstance(header, fits.header.Header):
        log.error("Header %s is not %s" %
                  (type(header), fits.header.Header))
        return False
    elif not isinstance(keywords, DataFrame):
        log.error("Keywords %s is not %s" % (type(keywords), DataFrame))
        return False
    return all([*map(lambda x:
                     validate_keyrow(header, x[1], dripconf=dripconf),
                     keywords.iterrows())])


def validate_file(filename, keywords, dripconf=False):
    """
    Validate filename header against keywords table

    Parameters
    ----------
    filename : str
        Path to a FITS file
    keywords : pandas.DataFrame
    dripconf : bool
        If True, will check the configuration file for the required
        parameter before checking headers (using getpar)

    Returns
    -------
    bool
        True if all header keywords were validated, False otherwise

    """
    if not isinstance(filename, str):
        log.error("Filename must be a string, received %s" % type(filename))
        return False
    if not isinstance(keywords, DataFrame):
        log.error("Keywords must be %s, received %s" %
                  (DataFrame, type(keywords)))
        return False
    if not os.path.isfile(filename):
        log.error("Not a file: %s" % filename)
        return False
    header = fits.getheader(filename)
    return validate_header(header, keywords, dripconf=dripconf)


def hdcheck(filelist, dohdcheck=None, dripconf=False, kwfile=None):
    """
    Checks file headers against validity criteria

    Checks if the headers of the input files satisfy criteria described
    in a separate file: input-key-definitions.txt, kept in the
    calibration data directory.  This file should have columns keyname,
    condition, type, enum, format, min, max.  The condition should be
    Boolean, indicating whether the keyword is required to be present.
    Type should be a string specifying the Python data type required
    for the value.  Enum indicates the specific values the keyword
    is allowed to have.  Format describes the format requirement for
    a string value.  Min and max indicate the minimum and maximum
    values allowed for a numerical value.  For any of these fields,
    a '.' indicates no requirement.

    Parameters
    ----------
    filelist : list or str
        File paths to check
    dohdcheck : str
        Configuration keyword to check.  If provided, this keyword will
        be read from sofia_redux.instruments.forcast.configuration;
        if set to 1, the headers will be checked.  Otherwise, the check
        will be aborted and this function will return True.
    dripconf : bool
        If True, will check the configuration file for the required
        parameter before checking headers (using getpar).
    kwfile : str
        Keyword definition file to use.  If not provided, the default
        file will be used (input-key-definition.yxy, in the calibration
        data directory.

    Returns
    -------
    bool
        True if headers of all files are correct; False otherwise
    """
    if isinstance(filelist, str):
        flist = [filelist]
    elif isinstance(filelist, Iterable):
        flist = []
        for item in filelist:
            flist.append(item)
            if not isinstance(item, str):
                log.error("Must specify file name(s) as strings: "
                          "recieved %s" % item)
                return False
    else:
        log.error(
            "hdcheck - must specify file name(s) as strings or list")
        return False

    keywords = hdrequirements(kwfile)
    allvalid = True
    for filepath in flist:
        if not os.path.isfile(filepath):
            log.error("File does not exist: %s" % filepath)
            allvalid = False
            continue

        try:
            header = fits.getheader(filepath)
        except OSError:
            header = None

        if not isinstance(header, fits.header.Header):
            log.error("Could not read FITS header: %s" % filepath)
            allvalid = False
            continue

        if isinstance(dohdcheck, str):
            do_check = getpar(header, dohdcheck,
                              comment='performed header checking?')
            if do_check != '1':
                log.info("%s set to %s" % (dohdcheck, do_check))
                log.info("Skipping keywords validation")
                return True

        header_ok = validate_header(
            header, DataFrame(keywords), dripconf=dripconf)
        allvalid &= header_ok
        if not header_ok:
            log.error("File has wrong header: %s" % filepath)

    return allvalid
