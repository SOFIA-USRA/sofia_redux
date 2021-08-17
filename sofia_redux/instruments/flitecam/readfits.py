# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
from astropy.wcs import WCS, SingularMatrixError
import numpy as np

from sofia_redux.toolkit.utilities.fits import hdinsert, robust_read

__all__ = ['readfits']


def readfits(filename):
    """
    Return the data array from the input file.

    Reads a specified raw FLITECAM FITS file and returns a
    new HDUList with the flux image in the primary extension
    with EXTNAME = FLUX.

    Raw FLITECAM data frequently has problems with WCS definition.
    This algorithm attempts to correct the most common WCS problems
    and return standardized WCS definition.  However, some raw
    files may be unfixable; in that case, an error is raised and
    the file must be manually fixed before proceeding.

    Parameters
    ----------
    filename : str
        File path of the file to be read.

    Returns
    -------
    astropy.io.fits.HDUList
        Output HDUList.

    Raises
    ------
    ValueError
        If the FITS header is unreadable.
    """
    data, header = robust_read(filename)
    if header is None:
        header = fits.header.Header()
    hdinsert(header, 'EXTNAME', 'FLUX', comment='Extension name')

    if data is None:
        log.warning(f"Error loading file {filename}")
        return

    # add a BUNIT: digital number
    # (will be modified to ct/s in the lincor module)
    hdinsert(header, 'BUNIT', 'ct', 'Data units')

    # fix a couple known bad old WCS keywords
    try:
        header['RADESYS'] = header['RADECSYS']
    except KeyError:
        pass
    bad_keys = ['XPIXELSZ', 'YPIXELSZ', 'RADECSYS']
    for bad_key in bad_keys:
        try:
            del header[bad_key]
        except KeyError:
            pass

    # replace CD matrix with CDELT/CROTA
    try:
        hwcs = WCS(header)
    except SingularMatrixError as werr:
        log.debug(f'Error from astropy.wcs: {werr}')
        err = 'FITS header is unreadable. Data reduction ' \
              'cannot continue.'
        log.error(err)
        raise ValueError(err) from None

    try:
        pixscal = np.sqrt(np.linalg.det(hwcs.wcs.cd))
    except AttributeError as werr:
        log.debug(f'Error from astropy.wcs: {werr}')
        err = 'FITS header is unreadable. Data reduction ' \
              'cannot continue.'
        log.error(err)
        raise ValueError(err) from None

    pc = hwcs.wcs.cd / pixscal
    angle = (360 - np.rad2deg(np.arccos(pc[0][0]))) % 360
    cd_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
    for key in cd_keys:
        del header[key]

    # make an astropy standard WCS
    header['CDELT1'] = pixscal
    header['CDELT2'] = pixscal
    header['CROTA2'] = angle
    hwcs = WCS(header)

    # remove all old WCS keys
    wcs_keys = ['CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
                'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2',
                'CDELT1', 'CDELT2', 'CROTA2', 'CUNIT1',
                'CUNIT2', 'WCSNAME', 'WCSAXES',
                'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
                'RADECSYS', 'RADESYS']

    # add in standard keys
    for key in wcs_keys:
        if key in header:
            del header[key]
    hwcs_header = hwcs.to_header()
    for key in hwcs_header:
        hdinsert(header, key, hwcs_header[key], hwcs_header.comments[key])

    # if necessary, replace astropy standard PC matrix
    # with CROTA2 for compatibility with other SOFIA WCS
    # standards
    pc_keys = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']
    for key in pc_keys:
        if key in header:  # pragma: no cover
            del header[key]
    hdinsert(header, 'CROTA2', angle, 'Coordinate system rotation angle')

    primary = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([primary])

    return hdul
