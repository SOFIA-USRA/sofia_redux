# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy import log
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from sofia_redux.instruments.forcast.calcvar import calcvar
from sofia_redux.toolkit.utilities.fits \
    import hdinsert, href, kref, robust_read

__all__ = ['addparent', 'readfits']


def addparent(name, header,
              comment="id or name of file used in the processing"):
    """
    Add an id or file name to a header as PARENTn

    Adds the ID or filename of an input file to a specified header array,
    under the keyword PARENTn, where n is some integer greater than 0.
    If a previous PARENTn keyword exists, n will be incremented to
    produce the new keyword.

    If no PARENTn keyword exists, a new card will be appended to the end
    of the header.  Otherwise, the card will be inserted after PARENT(n-1).

    Parameters
    ----------
    name : str
        Name or id of the file to be recorded in the header
    header : astropy.io.fits.header.Header
        Input header to be updated
    comment : str
        Comment for PARENTn

    Returns
    -------
    None
    """
    parents = header.cards['PARENT*']
    value = os.path.basename(name)
    if len(parents) == 0:
        hdinsert(header, 'PARENT1', value, comment=comment, refkey=kref)

    existing_values = [x[1] for x in parents]
    if value in existing_values:
        return

    existing_keys = [x[0] for x in parents]
    parentn = 1
    last_parent = kref
    while True:
        key = 'PARENT' + str(parentn)
        if key in existing_keys:
            last_parent = key
            parentn += 1
        else:  # pragma: no cover
            break
    hdinsert(header, key, value,
             refkey=last_parent, comment=comment, after=True)


def readfits(filename, update_header=None, key=None, variance=False,
             stddev=False, fitshead=False, fitshdul=False,
             comment=None, hdu=0):
    """
    Returns the array from the input file

    Reads and returns the image array from a specified FITS file.  If an
    input header is specified, it updates it with the filename or ID of
    the newly created read file (via addparent).

    Parameters
    ----------
    filename : str
        file path of the file to be read
    update_header : astropy.io.fits.header.Header, optional
        Header to update with a PARENTn keyword containing the file or
        ID of the new read file.
    key : str, optional
        If provided along with `header`, the ID of filename added to the
        PARENTn keyword in the header is also added under this keyword.
    variance : bool, optional
        If True, the variance associated with the input FITS file will
        be calculated and returned by appending an additional dimension
        to the output array i.e., (2, ...).  If fitshdul is True, the
        variance will be stored in a separate extension instead.
    stddev : bool, optional
        If True, the variance will be calculated, but its square root
        will be returned, as a standard deviation value, in place of
        the variance.
    fitshead : bool, optional
        if True, the output returned will be a 2-tuple where the first
        element is the data array, and the second will be the FITS header
        read from the file.
    fitshdul : bool, optional
        if True, the output returned will be an astropy HDUList.  Takes
        precedence over the fitshead key.
    comment : str, optional
        If set, will add a comment to `key` in the header
    hdu : int, optional
        Header Data Unit.  Default is 0 (primary)

    Returns
    -------
    numpy.ndarray, (numpy.ndarray, astropy.io.fits.header.Header),
    or astropy.io.fits.HDUList
        Image array or (Image array, header) if fitshead=True
        or astropy HDUList if fitshdul=True.
    """
    data, header = robust_read(filename, data_hdu=hdu, header_hdu=hdu)
    if header is None:
        header = fits.header.Header()
    hdinsert(header, 'EXTNAME', 'FLUX', comment='extension name')

    # Add some delineation to the header
    hdinsert(header, kref, 'Keyword reference',
             comment='Header reference keyword', refkey='HISTORY')
    hdinsert(header, href, 'History reference',
             comment='Header reference keyword', refkey='HISTORY',
             after=True)
    hdinsert(header, 'COMMENT',
             '--------------------------------------------', refkey=kref)
    hdinsert(header, 'COMMENT',
             '--------- Pipeline related Keywords --------', refkey=kref)
    hdinsert(header, 'COMMENT',
             '--------------------------------------------', refkey=kref)
    hdinsert(header, 'HISTORY',
             '---------------------------------------', refkey=href)
    hdinsert(header, 'HISTORY',
             '---------- PIPELINE HISTORY -----------', refkey=href)
    hdinsert(header, 'HISTORY',
             '---------------------------------------', refkey=href)

    if data is None:
        log.warning("error loading file %s" % filename)
        return

    if isinstance(update_header, fits.header.Header):
        obs_id = header.get('OBS_ID', os.path.basename(filename))
        if key is not None:
            hdinsert(update_header, key, obs_id,
                     comment=comment, refkey='HISTORY')
        addparent(obs_id, update_header, comment=comment)

    # add a BUNIT: digital number (will be converted to Me/s in stack step)
    hdinsert(header, 'BUNIT', 'ct', 'Data units', refkey=kref)

    if variance or stddev:
        dataout = np.empty([2] + list(data.shape), dtype=data.dtype)
        dataout[0] = data
        var = calcvar(data, header)
        if stddev:
            dataout[1] = np.sqrt(var) if var is not None else np.nan
        else:
            dataout[1] = var if var is not None else np.nan
    else:
        dataout = data

    if fitshdul:
        if variance or stddev:
            primary = fits.PrimaryHDU(data=dataout[0], header=header)

            # make a basic variance header from the WCS in the primary
            wcs = WCS(header)
            vhead = wcs.to_header(relax=True)
            if stddev:
                hdinsert(vhead, 'BUNIT', 'ct', 'Data units', refkey=kref)
                var = fits.ImageHDU(data=dataout[1],
                                    header=vhead, name='ERROR')
            else:
                hdinsert(vhead, 'BUNIT', 'ct2', 'Data units', refkey=kref)
                var = fits.ImageHDU(data=dataout[1],
                                    header=vhead, name='VARIANCE')
            hdul = fits.HDUList([primary, var])
        else:
            hdul = fits.HDUList(fits.PrimaryHDU(data=dataout, header=header))
        return hdul
    elif fitshead:
        return dataout, header
    else:
        return dataout
