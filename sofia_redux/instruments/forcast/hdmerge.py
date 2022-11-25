# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log

from sofia_redux.instruments import forcast as drip
from sofia_redux.toolkit.utilities.fits import merge_headers

__all__ = ['hdmerge']


def hdmerge(headers, reference_header=None):
    """
    Merge input headers.

    Keyword handling is defined in the configuration file at
    data/header_merge.cfg.

    Parameters
    ----------
    headers : list of astropy.io.fits.Header
        Headers to check.
    reference_header : astropy.io.fits.Header, optional
        If provided, is used as the base header to be updated.
        If not provided, the first header (by DATE-OBS) is used
        as the reference.
    Returns
    -------
    astropy.io.fits.Header
        Combined header.
    """
    data_path = os.path.join(os.path.dirname(drip.__file__), 'data')
    kwfile = os.path.join(data_path, 'header_merge.cfg')

    if not os.path.isfile(kwfile):
        msg = f'Config file {kwfile} is missing.'
        log.error(msg)
        raise IOError(msg)

    new_header = merge_headers(headers, kwfile,
                               reference_header=reference_header)

    return new_header
