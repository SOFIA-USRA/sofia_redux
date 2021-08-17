# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from sofia_redux.instruments.flitecam.calcvar import calcvar
from sofia_redux.toolkit.utilities.fits \
    import hdinsert, getdata, set_log_level

__all__ = ['lincor']


def _imgpoly(x, coeffs):
    """
    Evaluate an array of polynomials.

    Parameters
    ----------
    x : np.ndarray
        Independent values. Dimensions should match coeffs.shape[1:].
    coeffs : np.ndarray
        Polynomial coefficients.  First dimension is coefficient
        number, starting with the lowest order coefficient.
        Remaining dimensions should match the `x` array.

    Returns
    -------
    polyval : np.ndarray
        The polynomial array evaluated at x.  Matches dimensions
        of input array.
    """
    polyval = np.zeros_like(x)
    for j in range(coeffs.shape[0] - 1, -1, -1):
        c = coeffs[j]
        polyval *= x
        polyval += c
    return polyval


def _linearize(image, coeffs, itime, readtime, ndr):
    """
    Correct a FLITECAM image for linearity.

    Follows 2004PASP..116..352V.

    Parameters
    ----------
    image : np.ndarray
        Raw FLITECAM data, divided by DIVISOR value.
    coeffs : np.ndarray
        Linearity coefficients array.
    itime : float
        Integration time.
    readtime : float
        Readout time.
    ndr : int
        Number of non-destructive reads.

    Returns
    -------
    linearized : np.ndarray
        The linearity corrected image.
    """
    # Following mc_flitecamlincor, from fspextool
    c0 = coeffs[0]

    # flat pedestal: from lab data used to create coefficients
    # flat readout time = 197.256 msec (slowcnt=4)
    flat_tread = 0.197256
    flatped = c0 * flat_tread

    # correct iteratively
    niter = 2
    newimage = image
    for i in range(niter):
        # pedestal = ct/s * t_readout/itime * [(ndr+1)/2 - f], with f=0.5
        pedimage = newimage * readtime * float(ndr) / (2.0 * itime)
        finimage = image + pedimage

        # correct pedestal
        c_pedimage = pedimage.copy()
        idx = (pedimage - flatped >= 0)
        if np.any(idx):
            # use the fit only where the value is greater than
            # the flat pedestal
            corr = c0 / _imgpoly(pedimage - flatped, coeffs)
            corr[corr < 1.] = 1.

            cped = pedimage * corr
            c_pedimage[idx] = cped[idx]

        # correct final image similarly
        c_finimage = finimage.copy()
        idx = (finimage - flatped >= 0)
        if np.any(idx):
            corr = c0 / _imgpoly(finimage - flatped, coeffs)
            corr[corr < 1.] = 1.
            cfin = finimage * corr
            c_finimage[idx] = cfin[idx]

        newimage = c_finimage - c_pedimage

    return newimage


def lincor(hdul, linfile, saturation=None):
    """
    Correct input flux data for detector nonlinearity.

    The image is corrected by multiplying it by the factor
    1 / (1 + (a1/a0) * counts + ... + (an/a0) * counts^n), where the
    a values are the coefficients given in the input linearity file
    and the counts are the values given in the input image.

    The output flux is also divided by exposure time to convert from
    counts (ct) to flux units (ct/s).

    Parameters
    ----------
    hdul : fits.HDUList
        Input data.  Should have a single primary FLUX extension.
    linfile : str
        Path to an input FITS file containing linearity coefficients.
        The file should contain a single primary image extension,
        with dimensions 1024 x 1024 x n, giving the correction
        coefficients for each pixel in the FLITECAM array.
    saturation : float, optional
        If provided, values (as flux / divisor) above this level are
        marked as bad in the output BADMASK extension (0 = good, 1 = bad).

    Returns
    -------
    fits.HDUList
        Corrected data, with updated FLUX and additional ERROR
        and BADMASK extension.

    Raises
    ------
    ValueError
        If the linearity file is bad or missing.
    """
    # input data
    header = hdul[0].header
    data = hdul[0].data

    # read linfile
    coeffs = getdata(linfile)
    if coeffs is None:
        raise ValueError('Missing linearity file')
    if len(coeffs.shape) != 3 or coeffs.shape[1:] != data.shape:
        raise ValueError('Linearity file has wrong shape')

    # exposure keywords
    itime = header.get('ITIME', 0)
    coadds = header.get('COADDS', 0)
    ndr = header.get('NDR', 0)
    readtime = header.get('TABLE_MS', 0) / 1000.
    divisor = float(header.get('DIVISOR', 1.))

    # check for saturation
    mask = np.zeros(data.shape, dtype=np.int16)
    if saturation is not None and saturation > 0:
        saturated = ((data / divisor) > saturation)
        mask[saturated] = 1

    # correct the flux
    linearized = _linearize(data / divisor, coeffs, itime, readtime, ndr)
    linearized *= divisor

    # calculate error plane from linearized data
    var = calcvar(linearized, header)
    error = np.sqrt(var)

    # correct data for exposure time
    exptime = ndr * coadds * itime
    linearized /= exptime

    # update the primary BUNIT
    hdinsert(header, 'BUNIT', 'ct/s', 'Data units')

    # add linfile to header
    hdinsert(header, 'LINFILE', linfile, comment='Linearity file')

    # make the output HDUList
    primary = fits.PrimaryHDU(data=linearized, header=header)

    # make a basic extension header from the WCS in the primary
    with set_log_level('CRITICAL'):
        wcs = WCS(header)
    ehead = wcs.to_header(relax=True)
    hdinsert(ehead, 'BUNIT', 'ct/s', 'Data units')
    err_ext = fits.ImageHDU(data=error, header=ehead, name='ERROR')
    hdinsert(ehead, 'BUNIT', '', 'Data units')
    mask_ext = fits.ImageHDU(data=mask, header=ehead, name='BADMASK')

    corrected = fits.HDUList([primary, err_ext, mask_ext])

    return corrected
