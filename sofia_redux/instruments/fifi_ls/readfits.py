# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log

from sofia_redux.instruments.fifi_ls.make_header import make_header
from sofia_redux.toolkit.utilities import gethdul, hdinsert

__all__ = ['readfits']


def readfits(filename, checkheader=False):
    """
    Read a FIFI-LS FITS file and return data and header.

    Parameters
    ----------
    filename : str
        path to a FIFI-LS FITS file
    checkheader : bool, optional
        Check whether the header is valid

    Returns
    -------
    fits.HDUList
    """
    hdul = gethdul(filename, verbose=True)
    if hdul is None:
        return (None, False) if checkheader else None

    result = make_header(hdul[0].header, checkheader=checkheader)
    header, success = result if checkheader else (result, True)

    hdul[0].header = header
    inst = str(header.get('INSTRUME', 'UNKNOWN')).strip().upper()
    if inst != 'FIFI-LS':
        log.error("Not a FIFI-LS file: %s" % filename)
        return (None, False) if checkheader else None
    hdinsert(hdul[0].header, 'FILENAME', os.path.basename(filename),
             comment='File name', refkey='HISTORY')

    return (hdul, success) if checkheader else hdul
