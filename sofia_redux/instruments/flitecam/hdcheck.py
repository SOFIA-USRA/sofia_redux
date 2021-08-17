# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
import configobj

__all__ = ['validate_header', 'hdcheck']


def validate_header(header, keywords):
    """
    Validate all keywords in a header against the keywords requirement.

    Parameters
    ----------
    header : astropy.io.fits.Header
    keywords : configobj.ConfigObj
    dripconf : bool
        If True, will check the configuration file for the required
        parameter before checking headers (using getpar)

    Returns
    -------
    bool
        True if all keywords in the header were validated, False otherwise
    """
    # Check a few important mode keywords
    nodding = header.get('NODDING', False)
    dithering = header.get('DITHER', False)

    # Add any that are True to the requirement set
    req_set = ['*']
    if nodding:
        req_set.append('nodding')
    if dithering:
        req_set.append('dithering')

    # Flag to track overall validity
    valid = True

    # Loop through keywords, checking against requirements
    reqdict = keywords.dict()
    for key, req in reqdict.items():
        # Retrieve requirements
        try:
            req_category = str(req['requirement']).strip()
        except KeyError:
            req_category = '*'
        try:
            req_dtype = str(req['dtype']).strip()
        except KeyError:
            req_dtype = 'str'
        try:
            req_drange = req['drange']
        except KeyError:
            req_drange = None

        # Get type class corresponding to string
        if req_dtype == 'bool':
            req_dtype_class = bool
        elif req_dtype == 'int':
            req_dtype_class = int
        elif req_dtype == 'float':
            req_dtype_class = float
        else:
            req_dtype_class = str

        # Check if key is required for this data type
        if req_category not in req_set:
            continue

        # Retrieve value from header and/or config file
        val = header.get(key, None)
        valtype = type(val)
        stype = valtype.__name__

        # Check if required key is present
        if val is None:
            valid = False
            msg = f'Required keyword {key} not found'
            log.warning(msg)
            continue

        # Check if key matches required type
        if req_dtype in ['str', 'bool', 'int']:
            # Use exact type for str, bool, int
            if stype != req_dtype:
                valid = False
                msg = f'Required keyword {key} has wrong ' \
                      f'type {stype}; should be {req_dtype}'
                log.warning(msg)
                continue
        elif req_dtype == 'float':
            # Allow any number type for float types
            if stype not in ['float', 'int']:
                valid = False
                msg = f'Required keyword {key} has wrong ' \
                      f'type {stype}; should be {req_dtype}'
                log.warning(msg)
                continue

        # Check if value meets range requirements
        if req_drange is not None:
            # Check for enum first -- ignore any others if
            # present. May be used for strings, bools, or numerical
            # equality.
            if 'enum' in req_drange:
                enum = req_drange['enum']

                # Make into list if enum is a single value
                if type(enum) is not list:
                    enum = [enum]

                # Cast to data type
                if req_dtype == 'bool':
                    enum = [True if str(e).strip().lower() == 'true'
                            else False for e in enum]
                else:
                    try:
                        enum = [req_dtype_class(e) for e in enum]
                    except ValueError as error:
                        msg = f'Error in header configuration file for ' \
                              f'key {key}'
                        log.error(msg)
                        raise error

                # Case-insensitive comparison for strings
                if stype == 'str':
                    enum = [str(e).strip().upper() for e in enum]
                    if val.strip().upper() not in enum:
                        valid = False
                        msg = f'Required keyword {key} has wrong ' \
                              f'value {val}; should be in {enum}'
                        log.warning(msg)
                        continue
                else:
                    if val not in enum:
                        valid = False
                        msg = f'Required keyword {key} has wrong ' \
                              f'value {val}; should be in {enum}'
                        log.warning(msg)
                        continue

            # Check for a minimum requirement
            # (numerical value must be >= minimum)
            else:
                if 'min' in req_drange and stype in ['int', 'float']:
                    try:
                        minval = req_dtype_class(req_drange['min'])
                    except ValueError as error:
                        msg = f'Error in header configuration file for ' \
                              f'key {key}'
                        log.error(msg)
                        raise error
                    if val < minval:
                        valid = False
                        msg = f'Required keyword {key} has wrong ' \
                              f'value {val}; should be >= {minval}'
                        log.warning(msg)
                        continue

                # Check for a maximum requirement
                # (numerical value must be <= maximum)
                if 'max' in req_drange and stype in ['int', 'float']:
                    try:
                        maxval = req_dtype_class(req_drange['max'])
                    except ValueError as error:
                        msg = f'Error in header configuration file for ' \
                              f'key {key}'
                        log.error(msg)
                        raise error
                    if val > maxval:
                        valid = False
                        msg = f'Required keyword {key} has wrong ' \
                              f'value {val}; should be <= {maxval}'
                        log.warning(msg)
                        continue
    return valid


def hdcheck(headers, kwfile):
    """
    Checks file headers against validity criteria

    Checks if the headers of the input files satisfy criteria described
    in a separate file, kept in the calibration data directory.  This
    file should be a configuration file in INI format, specifying the
    keyword name, condition, data type, and any range requirements.

    Parameters
    ----------
    headers : list of astropy.io.fits.Header
        Headers to check.
    kwfile : str
        Keyword definition file to use.

    Returns
    -------
    bool
        True if headers of all files are correct; False otherwise
    """

    if os.path.isfile(kwfile):
        try:
            reqconf = configobj.ConfigObj(kwfile)
        except configobj.ConfigObjError as error:
            msg = 'Error while loading header configuration file.'
            log.error(msg)
            raise error
    else:
        msg = f'{kwfile} is invalid file name for header configuration'
        log.error(msg)
        raise IOError(msg)

    allvalid = True
    for header in headers:
        if not isinstance(header, fits.Header):
            log.error('Could not read FITS header')
            allvalid = False
            continue

        header_ok = validate_header(header, reqconf)
        allvalid &= header_ok
        if not header_ok:
            log.error(f"File has wrong header: "
                      f"{header.get('FILENAME', 'UNKNOWN')}")

    return allvalid
