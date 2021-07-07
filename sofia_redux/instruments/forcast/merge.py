# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap, hdinsert, kref

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
from sofia_redux.instruments.forcast.merge_centroid import merge_centroid
from sofia_redux.instruments.forcast.merge_correlation import merge_correlation
from sofia_redux.instruments.forcast.merge_shift import merge_shift
from sofia_redux.instruments.forcast.readmode import readmode
from sofia_redux.instruments.forcast.rotate import rotate

addhist = add_history_wrap('Merge')

__all__ = ['merge']


def merge(data, header, variance=None, normmap=None,
          strip_border=True, rotation_order=1, resize=True):
    """
    Merge positive and negative instances of the source in the images

    After chop/nod subtraction, there are typically positive and
    negative instances of the sources in the same frame, although
    the number of images depends on the chop/nod mode.  This function
    shifts, coadds, and normalizes these sources to provide a single
    image of the source with increased signal-to-noise.  The method
    for determining the shift is read from the configuration file.
    In the configuration file, if CORMERGE is set to CENTROID, then
    a centroiding algorithm is used to determine the shift.  If
    CORMERGE is XCOR, a cross-correlation algorithm is used.  If
    CORMERGE is HEADER, then header data is used to determine the
    shift.  If CORMERGE is NOSHIFT (or the calculated shift is
    greater than the MAXSHIFT parameter), then no shifting and
    coadding is attempted.  IF the centroiding algorithm is selected
    and it fails for any reason, then a header shift algorithm is
    used instead.  After the image is merged, it is rotated by the
    SKY_ANGL in the header and the WCS keywords are updated.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (nrow, ncol)
    header : astropy.io.fits.header.Header
        Input FITS header; will be updated with a HISTORY message
    variance : numpy.ndarray, optional
        Variance array (nrow, ncol) to update in parallel with the
        data array
    normmap : numpy.ndarray, optional
        Array (nrow, ncol) of normalization values for each pixel.
        The normalization value corresponds to the number of
        exposures in each pixel.
    strip_border : bool, optional
        If True, will strip off any unnecessary NaN padding at the
        edges of the image, after rotation.
    rotation_order : int, optional
        Order for spline interpolation when rotating
    resize : bool, optional
        If True, image will be resized as necessary during merge.

    Returns
    -------
    numpy.ndarray, np.ndarray
        Merged array (nrow, ncol)
        Propagated variance array (nrow, ncol)
    """
    # sanity checks and initial values
    if not isinstance(header, fits.header.Header):
        header = fits.header.Header()
        log.warning("invalid header")
        addhist(header, 'Created header')

    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        addhist(header, 'was not applied (Invalid data)')
        log.error("invalid data array")
        return

    # Get original CRPIX values
    orig_crpix1 = header.get('CRPIX1', 1.0)
    orig_crpix2 = header.get('CRPIX2', 1.0)

    # Set initial normalization map
    if isinstance(normmap, np.ndarray):
        # normmap is not critical, so we can resize
        if normmap.shape != data.shape:
            normmap.resize(data.shape, refcheck=False)
        normmap.fill(0)
    else:
        normmap = np.zeros_like(data)

    dovar = isinstance(variance, np.ndarray) and variance.shape == data.shape
    var = None if not dovar else variance.copy()
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning('invalid variance')

    maxshift = getpar(header, 'MAXSHIFT', dtype=int, default=120,
                      comment='maximum allowable merge pixel shift')
    mode = readmode(header)
    cormerge = getpar(header, 'CORMERGE', comment='merging algorithm',
                      dtype=str, default='unknown').upper().strip()
    slit = str(header.get('SLIT', 'UNKNOWN')).upper().strip()
    imaging = slit not in ['NONE', 'UNKNOWN']
    if mode == 'C2NC4' or imaging:
        cormerge = 'NOSHIFT'
    addhist(header, 'Merging method is %s' % cormerge)

    def header_shift():
        addhist(header, 'Shift algorithm uses headers')
        x = imgshift_header(header, dither=False)
        chopnod = [x[k] for k in ['chopx', 'chopy', 'nodx', 'nody']]
        return merge_shift(data, chopnod, header=header, variance=var,
                           normmap=normmap, resize=resize,
                           nmc=mode == 'NMC', maxshift=maxshift)
    # Merge
    if cormerge == 'HEADER':
        merged = header_shift()
        addhist(header, 'Shift algorithm uses header')
    elif cormerge == 'CENTROID':
        merged = merge_centroid(data, header, variance=var,
                                normmap=normmap, resize=resize)
        if merged is None:
            log.warning('centroid failed; falling back to header shifts')
            merged = header_shift()
        else:
            addhist(header, 'Shift algorithm uses centroid')
    elif cormerge == 'XCOR':
        addhist(header, 'Shift algorithm uses cross-correlation')
        merged = merge_correlation(data, header, variance=var, resize=resize,
                                   normmap=normmap, maxshift=maxshift)
    elif mode == 'NMC':
        addhist(header, 'Shift algorithm not applied for NMC mode')
        log.info("NMC mode with no shift; dividing by two and rotating")
        normmap[~np.isnan(data)] = 2
        if dovar:
            var /= 4
        merged = data / 2, var
    else:
        addhist(header, 'Shift algorithm not applied for %s mode' % mode)
        log.info("%s mode, no shift; will rotate only" % mode)
        normmap[~np.isnan(data)] = 1
        merged = data.copy(), var
    if merged is None or len(merged) != 2:
        addhist(header, 'Merging algorithm failed')
        log.error('merging algorithm failed')
        return

    # Rotate and strip
    skyangle = getpar(header, 'SKY_ANGL', dtype=float, default=None)
    rotangle = 180 - skyangle if skyangle is not None else 0
    addhist(header, 'Image rotation of %f' % rotangle)
    log.info("Image rotation of %f" % rotangle)

    try:
        center = [header['CRPIX1'] - 1, header['CRPIX2'] - 1]
    except (KeyError, ValueError):
        log.warning('No CRPIX found; rotating around center of image.')
        center = None

    d0 = merged[0].copy()
    merged = rotate(d0, rotangle, header=header, variance=merged[1],
                    order=rotation_order, center=center,
                    strip_border=strip_border)
    _, mask = rotate(d0, rotangle, order=rotation_order, missing=0,
                     variance=normmap, center=center,
                     strip_border=strip_border)
    # To pass back out of function
    normmap.resize(mask.shape, refcheck=False)
    normmap[:, :] = mask.copy()

    # Update WCS
    if 'CROTA2' in header and 'CRPIX1' in header and 'CRPIX2' in header:
        # Update the rotation angle.  At this stage it should be 0
        header['CROTA2'] = 0.0
        addhist(header, 'New CROTA2 after rotation is 0.0 degrees')

        # also store the cumulative change to CRPIX
        dcrpix1 = header.get('DCRPIX1', 0.0)
        dcrpix2 = header.get('DCRPIX2', 0.0)
        hdinsert(header, 'DCRPIX1', dcrpix1 + header['CRPIX1'] - orig_crpix1,
                 comment='Change in CRPIX before registration', refkey=kref)
        hdinsert(header, 'DCRPIX2', dcrpix2 + header['CRPIX2'] - orig_crpix2,
                 comment='Change in CRPIX before registration', refkey=kref)
    else:
        addhist(header, 'CRPIX1, CRPIX2 or CROTA2 are not in header')

    # Update the product type
    hdinsert(header, 'PRODTYPE', 'MERGED',
             comment='product type', refkey=kref)

    return merged
