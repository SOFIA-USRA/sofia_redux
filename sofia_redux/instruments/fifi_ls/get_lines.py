# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
import pandas

from sofia_redux.instruments import fifi_ls
from sofia_redux.toolkit.utilities import goodfile

__all__ = ['get_lines']


def get_lines():
    """
    Retrieve FIFI-LS lines of interest.

    Requires primary_lines.txt file in fifi_ls/data/line_lists.
    This file must have 2 columns: wavelength (um), and name.

    Returns
    -------
    list of float, list of string
        Wavelengths and names for lines of interest.
    """
    linefile = os.path.join(os.path.dirname(fifi_ls.__file__), 'data',
                            'line_lists', 'primary_lines.txt')

    if not goodfile(linefile, verbose=True):
        msg = "Cannot read line list file: %s" % linefile
        log.error(msg)
        raise ValueError(msg)

    names = ['wavelength', 'name']
    df = pandas.read_csv(linefile, comment='#', names=names,
                         delim_whitespace=True)
    return list(df['wavelength']), list(df['name'])
