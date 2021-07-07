# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits

import sofia_redux.instruments.forcast.configuration as dripconfig

configuration = None

__all__ = ['getpar']


def getpar(header, parname, writename=None, dtype=None,
           comment=None, update_header=True, default=None,
           dripconf=True, warn=False):
    """
    Get a header or configuration parameter

    Looks up a parameter from the DRIP configuration or the provided
    header.  Values in the configuration file take precedence over
    values in the header.  This function converts all parameters to
    string values before returning them.  If a parameter cannot be
    found in either the configuration file or the header, then
    None will be returned.

    Any string values enclosed within single quotes will have those
    outer quotes removed.

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        Input header
    parname : str
        Parameter name
    writename : str
        If set, the parameter to be returned will be added or
        modified in the header
    update_header : bool
        Will update the header if True
    comment : str
        Comment to write to the header along with the parameter
    dtype : type
        Attempt to convert the value to a float or int type.
        Failure will return zero of either type.
    default
        Value to return if not found.  Note that an attempt will
        be made to convert default to type `dtype` if `dtype` is
        supplied.
    dripconf : bool, optional
        Indicates whether configuration values can overwrite header
        keyword values.
    warn : bool, optional
        Sends a message to logger at warning level if the default
        value is used.

    Returns
    -------
    String or None if from the configuration.  Otherwise it could
    be anything, or set it yourself via dtype.
    """

    def bad_exit():
        dout = default
        if dtype is not None:
            try:
                dout = dtype(default)
            except (ValueError, TypeError):
                dout = default
        if warn:
            log.warning("%s not found - default is %s" % (parname, dout))
        return dout

    if not isinstance(parname, str):
        log.error('invalid parname: %s (%s)' % (parname, type(parname)))
        return bad_exit()
    if header is None:
        header = fits.header.Header()
    if not isinstance(header, fits.header.Header):
        log.error('invalid header type (%s)' % type(header))
        return bad_exit()

    if not dripconf:
        config_val = {}
    else:
        if dripconfig.configuration is None:
            dripconfig.load()
        global configuration
        configuration = dripconfig.configuration
        config_val = configuration

    key = parname.strip().upper()
    for ckey in config_val.keys():
        if ckey.strip().upper() == key:
            value = config_val[ckey]
            from_header = False
            break
    else:
        value = header.get(key[:8])
        from_header = True

    # DETCHAN weirdness fixed here
    if key == 'DETCHAN':
        if str(value).strip().upper() in ['1', 'LW']:
            new_value = 'LW'
        else:
            new_value = 'SW'
        if new_value != value:
            value = new_value
            from_header = False

    if value is None:
        return bad_exit()
    elif isinstance(value, list):
        value = '[%s]' % ','.join([str(v) for v in value])

    # convert to dtype before updating header
    if dtype is not None:
        try:
            value = dtype(value)
        except (ValueError, TypeError):
            log.warning('Could not convert value to %s' % dtype)
            value = None

    header_update = update_header
    header_update &= (comment is not None) | (not from_header)

    if header_update and value is not None:
        setname = writename if isinstance(writename, str) else key
        setname = setname.strip().upper()[:8]
        if setname not in header:
            if 'HISTORY' in header:
                header.insert('HISTORY', (setname, value, comment))
            else:
                header[setname] = value, comment
        else:
            if comment is None:
                update_comment = header.comments[setname]
            else:
                update_comment = comment
            header[setname] = value, update_comment

    return value
