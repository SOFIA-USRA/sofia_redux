# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.instruments.forcast.getpar import getpar

__all__ = ['check_readout_shift']


def check_readout_shift(data, header):
    """
    Check data for 16 pixel shift

    Looks at specific bad pixel regions in raw data to try to detect
    16 pixel shift.  Procedure does not seem to work for XD or images
    of the slit because the background is too low for bad pixels to
    show contrast using STDDEV.

    Parameters
    ----------
    data : np.ndarray
    header : astropy.io.fits.header.Header

    Returns
    -------
    True if pixel shift is likely, False otherwise
    """
    if not isinstance(data, np.ndarray):
        log.error("invalid data type: %s" % type(data))
        return
    elif not isinstance(header, fits.header.Header):
        log.error("invalid header type: %s" % type(header))
        return

    if len(data.shape) == 3:
        p1 = data[0]
    elif len(data.shape) == 2:
        p1 = data
    else:
        log.error("invalid data shape %s" % repr(data.shape))
        return

    # bad pixel regions to test (x1, x2, y1, y2)
    sreg = [0, 2, 36, 75]
    lref = [47, 61, 178, 189]
    detchan = getpar(header, 'DETCHAN', default=None, dtype=str,
                     update_header=False, dripconf=False)
    reg = lref if detchan == 'LW' else sreg

    # Always return False for cross-dispersed
    spectel1 = header.get('SPECTEL1', '').strip().upper()
    spectel2 = header.get('SPECTEL2', '').strip().upper()
    spectel = spectel2 if detchan == 'LW' else spectel1
    if spectel in ['FOR_XG063', 'FOR_XG111']:
        return False

    # Always return False for slit image
    slit = header.get('SLIT', '').strip().upper()
    spec_opt = ['FOR_XG063', 'FOR_XG111', 'FOR_G063', 'FOR_G111',
                'FOR_G227', 'FOR_G329']
    if spectel not in spec_opt and slit not in ['NONE', 'UNKNOWN']:
        log.debug("Check readout shift: Returning False for "
                  "slit, spectel = %s, %s" % (slit, spectel))
        return False

    # Otherwise test bad pixel regions
    test1 = p1[reg[2]:reg[3], reg[0]:reg[1]]
    test2 = p1[reg[2]:reg[3], (reg[0] + 16):(reg[1] + 16)]

    # Use stdev insstead of mean for check -- test shows stdev
    # typically higher (10x in imaging, much less in spectra) in
    # region with bad pixels (so use 3X as check)
    badfac = 3.
    if np.nanstd(test2) > (badfac * np.nanstd(test1)):
        return True
    else:
        return False
