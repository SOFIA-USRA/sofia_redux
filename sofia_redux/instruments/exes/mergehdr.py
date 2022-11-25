# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log

from sofia_redux.instruments import exes
from sofia_redux.instruments.exes.utils import set_elapsed_time
from sofia_redux.toolkit.utilities.fits import merge_headers

__all__ = ['mergehdr']


def mergehdr(headers, reference_header=None):
    """
    Merge input headers.

    Keyword handling is defined in the configuration file at
    data/header_merge.cfg.

    Parameters
    ----------
    headers : list of fits.Header
        Headers to check.
    reference_header : fits.Header, optional
        If provided, is used as the base header to be updated.
        If not provided, the first header (by DATE-OBS) is used
        as the reference.

    Returns
    -------
    fits.Header
        Combined header.
    """
    data_path = os.path.join(os.path.dirname(exes.__file__),
                             'data', 'header')
    kwfile = os.path.join(data_path, 'header_merge.cfg')

    if not os.path.isfile(kwfile):
        msg = f'Config file {kwfile} is missing.'
        log.error(msg)
        raise IOError(msg)

    new_header = merge_headers(headers, kwfile,
                               reference_header=reference_header)

    # in addition, attempt to add/update elapsed time from start/end keys
    set_elapsed_time(new_header)

    return new_header
