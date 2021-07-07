# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""DRIP configuration"""

import os

from astropy import log
import configobj

import sofia_redux.instruments.forcast as drip

configuration = None
configuration_file = None
keywords = None
keyword_file = None

__all__ = ['load']


def load(config_file=None, quiet=False):
    """
    Load the DRIP configuration file

    We will also check the $DRIPCONF_CURRENT environment
    variable if no configuration file was supplied.  If
    that fails we look for a drip configuration file in
    the current directory.

    Parameters
    ----------
    config_file : str
        Path to the DRIP configuration file
    quiet : bool, optional
        Do not generate log messages

    Returns
    -------
    None
    """

    drip_path = os.path.dirname(drip.__file__)
    default_path = os.path.join(drip_path, 'data')
    config_name = 'dripconf.txt'
    global configuration, configuration_file

    if config_file is None:
        if os.path.isfile(config_name):
            config_file = config_name
        else:
            config_file = os.path.join(default_path, config_name)

    if isinstance(config_file, str):
        configuration_file = os.path.realpath(config_file)
        if not os.path.isfile(configuration_file):
            log.warning("DRIP configuration file does not exist: %s" %
                        configuration_file)
            configuration = None
        else:
            if not quiet:
                log.info("loading DRIP configuration file: %s" %
                         configuration_file)
            try:
                configuration = configobj.ConfigObj(config_file,
                                                    interpolation=False)
            except configobj.ParseError:
                configuration = None
            if not isinstance(configuration, configobj.ConfigObj):
                log.warning("Could not load DRIP configuration file")
                configuration = None
