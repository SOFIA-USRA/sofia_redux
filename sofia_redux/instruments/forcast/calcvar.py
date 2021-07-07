# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import hdinsert, href, add_history_wrap

from sofia_redux.instruments.forcast.getpar import getpar

addhist = add_history_wrap('Calcvar')

__all__ = ['calcvar']


def calcvar(data, header):
    """
    Calculate read and poisson noise of the variance from raw FORCAST images

    Calculates the read noise and poisson noise components of the variance
    from raw FORCAST images.  The following equation is used to derive the
    variance for each pixel:

        V = N * betaG/g + RN^2/g^2

    where V is the variance, N is the raw ADU in each pixel, betaG is
    the excess noise factor, g is the fain, and RN is the read noise
    in electrons.  g comes from the EPERADU keyword in the FITS header.
    RN is determined by the instrument team, but its value depends on
    the capacitance setting (recorded in the ILOWCAP keyword in the FITS
    header).  betaG is also determined by the instrument team.

    Parameters
    ----------
    data : numpy.ndarray
        Raw image array
    header : astropy.io.fits.header.Header
        Raw image header; will be updated with HISTORY message and keywords

    Returns
    -------
    numpy.ndarray
        Image array with the same dimensions as the input raw image
        containing the calculated variance.
    """
    if not isinstance(data, np.ndarray) or len(data.shape) not in [2, 3]:
        addhist(header, 'Did not calculate variance (Invalid data)')
        log.error("must provide valid data array")
        return

    if not isinstance(header, fits.header.Header):
        log.error("must provide valid header")
        return

    rn_high = getpar(header, 'RN_HIGH', dtype=float, default=2400.,
                     comment="Read noise for high capacitance mode")
    rn_low = getpar(header, 'RN_LOW', dtype=float, default=244.8,
                    comment="Read noise for low capacitance mode")
    beta_g = getpar(header, 'BETA_G', dtype=float, default=1.0,
                    comment="Excess noise")
    eperadu = getpar(header, 'EPERADU', dtype=float, default=136)
    ilowcap = getpar(header, 'ILOWCAP', dtype=int, default=1)
    rn = rn_low if ilowcap else rn_high

    detitime = getpar(header, 'DETITIME', dtype=float, default=-1)
    if detitime < 0:
        log.error("Missing detector integration time (DETITIME)")
        hdinsert(header, 'HISTORY',
                 "Did not calculate variance (Invalid header)", refkey=href)
        return
    integ = detitime / 2.0

    frmrate = getpar(header, 'FRMRATE', dtype=float, default=-1)
    if frmrate < 0:
        log.error("Missing frame rate (FRMRATE)")
        hdinsert(header, 'HISTORY',
                 "Did not calculate variance (Invalid header)", refkey=href)
        return

    history = ['Read noise is %s' % rn,
               'Excess noise factor is %s' % beta_g,
               'Gain is %s' % eperadu,
               'Integration time is %s' % integ,
               'Variance for raw data is calculate from',
               'V = N*betaG/(FR*t*g) + RN^2/(FR*t*g^2), where',
               'N is the raw ADU per frame in each pixel,',
               'betaG is the excess noise factor,',
               'FR is the frame rate, t is the integration time,',
               'g is the gain, and RN is the ',
               'read noise in electrons']

    # For older data DETITIME did not account for chop efficiency.
    # These data can be distinguished by the unpopulated NODTIME keyword
    nodtime = getpar(header, 'NODTIME', update_header=False, default=-9999,
                     dtype=int)
    if nodtime == -9999:
        history.extend(["integration time may be",
                        "overestimated for early FORCAST data"])

    for line in history:
        hdinsert(header, 'HISTORY', line, refkey=href)

    f1 = beta_g / (frmrate * integ * eperadu)
    f2 = ((rn / eperadu) ** 2) / (frmrate * integ)
    return (data * f1) + f2
