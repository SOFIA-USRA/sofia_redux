# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log

import sofia_redux.instruments.forcast.configuration as dripconfig

__all__ = ['setpar']


def setpar(parname, value):
    """
    Set a drip configuration parameter

    Sets a parameter value in a loaded configuration object.  This is
    intended to override the default values set by the configuration
    file.  The value provided should be the string representation of
    the value.

    The parameter name will be converted to lowercase in the\
    configuration

    Parameters
    ----------
    parname : str
        Keyword to set the value for
    value : str
        Value to set

    Returns
    -------
    None
    """
    if dripconfig.configuration is None:
        dripconfig.load()

    if not isinstance(parname, str):
        log.warning("setpar - invalid parname: %s (%s)" %
                    (parname, type(parname)))
        return
    if not isinstance(value, str):
        log.warning("setpar - invalid parameter value %s=%s (%s)" %
                    (parname, value, type(value)))
        return

    pname = parname.strip().lower()
    dripconfig.configuration[pname] = value
