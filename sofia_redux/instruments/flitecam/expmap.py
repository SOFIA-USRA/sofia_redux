# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import hdinsert

__all__ = ['expmap']


def expmap(hdul):
    """
    Append an exposure map matching the FLUX extension.

    Parameters
    ----------
    hdul : fits.HDUList
        Input data.  Should have FLUX, ERROR, and BADMASK extensions.

    Returns
    -------
    fits.HDUList
        Data with EXPOSURE extension attached.
    """
    updated = hdul.copy()

    header = hdul[0].header
    flux = hdul['FLUX'].data
    ehead = hdul['ERROR'].header.copy()

    expmap = np.full(flux.shape, header.get('EXPTIME', 0.0), dtype=float)

    hdinsert(ehead, 'BUNIT', 's', 'Data units')
    updated.append(fits.ImageHDU(data=expmap, header=ehead, name='EXPOSURE'))

    return updated
