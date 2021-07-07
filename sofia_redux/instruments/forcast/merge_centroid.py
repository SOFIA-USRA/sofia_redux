# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap, hdinsert, kref

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.imgshift_header import imgshift_header
from sofia_redux.instruments.forcast.peakfind import peakfind
from sofia_redux.instruments.forcast.readmode import readmode
from sofia_redux.instruments.forcast.shift import shift

addhist = add_history_wrap('Merge')

__all__ = ['merge_centroid']


def merge_centroid(data, header, variance=None, normmap=None, resize=True):
    """
    Merge an image using a centroid algorithm

    Add each frame of the data to a 2-d summation frame in a manner
    appropriate to the current reduction scheme.  Averaged by the
    number of frames.

    Parameters
    ----------
    data : numpy.ndarray
        data to be merged (nrow, ncol). i.e. frame with target images
    header : astropy.fits.header.Header
        FITS header of input data
    variance : numpy.ndarray, optional
        propagate provided variance (nrow, ncol)
    normmap : numpy.ndarray, optional
        The normalization map.  Each pixel contains an integer of the
        number of peaks used to create an average for the output array

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        - The merged image i.e. frame with images of object at chop and
          nod positions merged. (nrow, ncol)
        - The propagated variance array (nrow, ncol)
    """
    log.info('Using centroid to merge chop/nod frames')
    if not isinstance(header, fits.header.Header):
        log.error("invalid header")
        return

    var = variance.copy() if isinstance(variance, np.ndarray) else None
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        addhist(header, "chop positions not applied (invalid data)")
        log.error("invalid data")
        return
    elif np.isnan(data).all():
        addhist(header, "merge not applied (invalid data)")
        log.error("data are all NaN")
        return

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var
    if variance is not None and not dovar:
        addhist(header, 'Not propagating variance (Invalid variance)')
        log.warning("invalid variance")

    # Set initial normalization map
    if isinstance(normmap, np.ndarray):
        # normmap is not critical, so we can resize
        if normmap.shape != data.shape:
            normmap.resize(data.shape, refcheck=False)
        normmap.fill(0)
    else:
        normmap = np.zeros_like(data)
    normmap[~np.isnan(data)] = 1

    # Check the mode and return data if merge is not required for the
    # specific mode
    mode = readmode(header)
    if mode == 'NMC':
        npeaks = 3
    elif mode == 'NPC':
        npeaks = 4
    else:
        log.warning("%s mode is not recognized. No merging performed" % mode)
        return

    _ = getpar(header, 'MTHRESH', dtype=float, default=15.0,
               comment='threshold for peakfind at merge')
    shift_order = getpar(header, 'SHIFTORD', dtype=int, default=0,
                         comment='interpolate order for coadd/merge')
    border = getpar(
        header, 'BORDER', dtype=int, default=128,
        comment='additional border pixels')
    addhist(header, 'Shift interpolation order is %i' % shift_order)

    # Calculate chop and nod distances
    imgshift = imgshift_header(header, dither=False)
    distchop = np.sqrt(imgshift['chopx'] ** 2 + imgshift['chopy'] ** 2)
    distnod = np.sqrt(imgshift['nodx'] ** 2 + imgshift['nody'] ** 2)

    # Find peak positions of the instances of the stars in the array
    find_img = np.zeros_like(data)
    clip = border + 10
    find_img[clip: -clip, clip: -clip] = data[clip: -clip, clip: -clip]

    fwhm = getpar(header, 'MFWHM', dtype=float, default=-1,
                  comment="fwhm used for peakfind at merge")
    if fwhm < 0:
        log.info("using default FWHM in peakfind")
        fwhm = None
    else:
        log.info("FWHM is %s in peakfind" % fwhm)

    kwargs = {'npeaks': npeaks,
              'chopnoddist': [distchop, distnod],
              'coordinates': True}
    if fwhm is not None:
        kwargs['fwhm'] = fwhm
    found = peakfind(find_img, **kwargs)

    nfound = len(found)
    if nfound != npeaks:
        log.warning("wrong number of peaks found")
        return

    fluxes = []
    for x, y in found:
        try:
            fluxes.append(find_img[int(y), int(x)])
        except IndexError:
            pass
    fluxes = np.array(fluxes)
    npos = (fluxes > 0).sum()
    nneg = (fluxes < 0).sum()

    ok = True
    if mode == 'NMC' and (npos != 1 or nneg != 2):
        ok = False
    elif mode == 'NPC' and (npos != 2 or nneg != 2):
        ok = False
    if not ok:
        log.warning("wrong peaks found for %s mode" % mode)
        return

    # Find the peak position closer to the CRPIX
    # (or the brightest source, if recorded) and store it in
    # base_coords
    crx = getpar(header, 'CRPIX1', dtype=int, default=data.shape[1] / 2)
    cry = getpar(header, 'CRPIX2', dtype=int, default=data.shape[0] / 2)
    cx = getpar(header, 'SRCPOSX', dtype=int, default=crx)
    cy = getpar(header, 'SRCPOSY', dtype=int, default=cry)
    dr = [(v[0] - cx) ** 2 + (v[1] - cy) ** 2 for v in found]
    base_idx = np.array(dr).argmin()
    base_coords = np.array(found[base_idx])
    xyoffsets = base_coords - np.array(found)

    # Arrays for accumulated and original data that may or may not be resized
    daccum = np.zeros_like(data)
    naccum = np.zeros(data.shape)
    vaccum = np.zeros_like(var) if dovar else None
    nans = np.isnan(data)
    data0 = data.copy()
    data0[nans] = 0
    norm0 = (~nans).astype(float)
    if dovar:
        var0 = var.copy()
        var0[nans] = 0
    else:
        var0 = None

    # The idea here is that for each found peak, we check if the
    # instance of the star is positive or negative.  Then, we shift
    # the image to match the reference star (the one closer to the
    # center) and we add them so the two instances are positive
    for idx, (xyoff, xypeak) in enumerate(zip(xyoffsets, found)):
        sign = 1 if data[int(xypeak[1]), int(xypeak[0])] >= 0 else -1
        addhist(header, 'X, Y shifts are %.3f,%.3f for peak %.3f,%.3f' %
                (xyoff[0], xyoff[1], xypeak[0], xypeak[1]))
        hdinsert(header, 'MRGX%i' % idx, xypeak[0], refkey=kref,
                 comment='X coordinates during merge')
        hdinsert(header, 'MRGY%i' % idx, xypeak[1], refkey=kref,
                 comment='Y coordinates during merge')
        hdinsert(header, 'MRGDX%i' % idx, xyoff[0], refkey=kref,
                 comment='X shift during merge')
        hdinsert(header, 'MRGDY%i' % idx, xyoff[1], refkey=kref,
                 comment='Y shift during merge')

        # Create an instance of the input image shifted so the
        # current star position matches the base_coord values
        # replace NaNs with zero for adding
        shiftd, _ = shift(data0, xyoff, order=shift_order, missing=0,
                          resize=resize)
        shiftn, shiftv = shift(norm0, xyoff, order=0, variance=var0,
                               resize=resize, missing=0)

        if resize:  # we need to update the size for addition later
            # - Resize original data; note header is updated here
            data0, _ = shift(data0, xyoff, order=shift_order, missing=0,
                             resize=True, header=header, no_shift=True)
            norm0, var0 = shift(norm0, xyoff, order=0, variance=var0,
                                resize=True, missing=0, no_shift=True)

            # - Resize accumulated data
            daccum, _ = shift(daccum, xyoff, order=shift_order, missing=0,
                              resize=True, no_shift=True)
            naccum, vaccum = shift(naccum, xyoff, order=0, variance=vaccum,
                                   resize=True, missing=0, no_shift=True)

        daccum += sign * shiftd
        naccum += shiftn
        if dovar:
            vaccum += shiftv

    # Put NaNs back in
    daccum[naccum == 0] = np.nan
    if vaccum is not None:
        vaccum[naccum == 0] = np.nan

    # Normalize the merged image and variance by the merge mask
    # Add one for NMC mode for the doubled source
    idx = naccum != 0
    if mode == 'NMC':
        naccum[idx] += 1
    daccum[idx] /= naccum[idx]
    if vaccum is not None:
        vaccum[idx] /= naccum[idx] ** 2

    # if we resized there is a possibility of a NaN data based on
    # the distribution of sources in the image - cut it off.
    if resize:
        nanrows = np.all(np.isnan(daccum), axis=1)
        nancols = np.all(np.isnan(daccum), axis=0)
        xl, xu = np.argmax(~nancols), np.argmax(np.flip(~nancols))
        yl, yu = np.argmax(~nanrows), np.argmax(np.flip(~nanrows))
        xu, yu = len(nancols) - xu, len(nanrows) - yu
        daccum = daccum[yl: yu, xl: xu]
        naccum = naccum[yl: yu, xl: xu]
        vaccum = vaccum[yl: yu, xl: xu] if dovar else None
        shape = daccum.shape
        # update the header
        header['NAXIS1'] = shape[1]
        header['NAXIS2'] = shape[0]
        if 'CRPIX1' in header:
            header['CRPIX1'] -= xl
        if 'CRPIX2' in header:
            header['CRPIX2'] -= yl
        if 'SRCPOSX' in header:
            header['SRCPOSX'] -= xl
        if 'SRCPOSY' in header:
            header['SRCPOSY'] -= yl

    # To pass back out of function
    normmap.resize(naccum.shape, refcheck=False)
    normmap[:, :] = naccum.copy()

    return daccum, vaccum
