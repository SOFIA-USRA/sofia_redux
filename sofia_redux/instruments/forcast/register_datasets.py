# Licensed under a 3-clause BSD style license - see LICENSE.rst

from warnings import catch_warnings, simplefilter

from astropy import log
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap, hdinsert

from sofia_redux.instruments.forcast.getpar import getpar
from sofia_redux.instruments.forcast.register import register
from sofia_redux.instruments.forcast.shift import shift

addhist = add_history_wrap('Coadd')

__all__ = ['wcs_shift', 'get_shifts', 'expand_array', 'shift_set',
           'shift_datasets', 'resize_datasets', 'register_datasets']


def wcs_shift(header, refheader, xy, wcskey=' '):
    """
    Return the pixel offset of a point on header relative to refheader

    Parameters
    ----------
    header : astropy.io.fits.header.Header

    refheader : astropy.io.fits.header. Header
    xy : array_like of float
        2-element (x, y) coordinate to shift or (n, (x, y)) coordinates

    Returns
    -------
    numpy.ndarray
        (x, y) offset in pixels of xy on header relative to refheader.
        This is the shift you would need to apply to xy in order for
        that point to align with that same point on the reference
        header after transformation.
    """
    wcs_in = WCS(header, key=wcskey)
    wcs_out = WCS(refheader, key=wcskey)
    xyin = np.array(xy)
    if len(xyin.shape) == 1:
        xyin = np.array([xyin])

    wxy = wcs_in.wcs_pix2world(xyin, 1)
    xyin_on_ref = wcs_out.wcs_world2pix(wxy, 1)
    return xyin - xyin_on_ref


def get_shifts(datasets, user_shifts=None, refset=0, basehead=None,
               algorithm=None, initial_reference=None,
               do_wcs_shift=False, wcskey=' '):
    """
    Returns all shifts relative to the reference set.

    No error check is performed.  Should be performed by the calling
    function.

    Parameters
    ----------
    datasets : array_like of tuples
        All elements are 3-tuples.  Each tuple represents a data set consisting
        where the elements are as follows:

            1. numpy.ndarray
                image array (nrow, ncol).  Image shapes are allowed to
                differ from others in the inputs data set.
            2. astropy.io.fits.header.Header
                FITS header for the image array
            3. numpy.ndarray, optional
                variance array (nrow, ncol).  Used to propagate variance.
                Must be the same shape as the data array in the first
                element.
            4. numpy.ndarray, optional
                normalization map (nrow, ncol) to propagate.  Must be the
                same shape as the data array in the first element.

    refset : int, optional
        Index of the dataset to use as the reference.  Default is the first
        set in the list (0).
    user_shifts : array_like of array_like, optional
        User shifts if required.  No shift may be represented by None.
        All other shifts should be supplied as (dx, dy).  Length must be
        the same length as datasets
    basehead : astropy.io.fits.header.Header, optional
        FITS header to update with HISTORY messages
    algorithm : str, optional
        Registration and coadding algorithm.  If None, defaults to the
        coadding algorithm determined by the drip configuration or
        reference header.
    initial_reference : array_like, optional
        An optional reference to pass if an initial shift on the reference
        dataset is required.
    do_wcs_shift : bool, optional
        If set, offsets returned are intended to be added to the CRPIX
        values in the header, rather than applied to the image array.
    wcskey : str, optional
        If not ' ', an alternate WCS is used for calculating WCS shifts.
        For spectral images, it is expected that wcskey='A'.

    Returns
    -------
    numpy.ndarray
        (nsets, (x, y)) array of offsets
    """
    if basehead is None:
        basehead = fits.header.Header()
        addhist(basehead, 'Created header')
    nsets = len(datasets)
    positions = [None] * nsets if user_shifts is None else user_shifts
    refdata = datasets[refset][0].copy()
    refheader = datasets[refset][1].copy()
    refposition = positions[refset]

    if algorithm is None:
        algorithm = getpar(refheader, 'CORCOADD')

    if algorithm == 'OVERRIDE':
        # just returns shifts, for either WCS shifts or image shifts
        shifts = []
        for (dset, pos) in zip(datasets, positions):
            init = [0, 0] if initial_reference is None else initial_reference
            shifts.append(register(dset[0], dset[1].copy(), reference=init,
                                   position=pos, get_offsets=True))
        return np.array(shifts)

    # shift the reference if necessary
    reference = refdata.copy()
    shift_init = np.array([0., 0.])
    if algorithm == 'HEADER' or initial_reference is not None:
        reference = register(
            refdata, refheader.copy(), reference=initial_reference,
            position=refposition, crpix=shift_init, algorithm=algorithm)
    else:
        hdinsert(basehead, 'HISTORY', "First image; not shifted", after=True)

    # non-reference sets
    sets = [s for (i, s) in enumerate(datasets) if i != refset]
    positions = [s for (i, s) in enumerate(positions) if i != refset]

    # for alternate WCS (spectral case)
    strip_key = wcskey.strip()
    if strip_key != '':
        refxy = [refheader[f'CRPIX1{strip_key}'],
                 refheader[f'CRPIX2{strip_key}'],
                 refheader[f'CRPIX3{strip_key}']]
    else:
        refxy = [refheader['CRPIX1'],
                 refheader['CRPIX2']]

    s0 = shift_init[0], shift_init[1]
    shifts = []
    for dset, pos in zip(sets, positions):
        if algorithm == 'WCS':
            wcs_xy = wcs_shift(refheader, dset[1], refxy,
                               wcskey=wcskey)[0]
            if len(wcs_xy) == 3:
                # coords are wavelength, dec, ra
                wcs_xy = np.array([wcs_xy[2], wcs_xy[1]])
            shifts.append(wcs_xy)
        else:
            s = register(dset[0], dset[1].copy(),
                         reference=reference,
                         algorithm=algorithm,
                         position=pos, get_offsets=True)
            shifts.append(s)
    result = shifts
    result.insert(refset, s0)

    if do_wcs_shift:
        crpix_shifts = result.copy()
        for i, s in enumerate(result):
            # skip if shift failed
            if s is not None:
                dset = datasets[i]

                # subtract out wcs shift to get crpix movement
                wcs_xy = wcs_shift(refheader, dset[1], refxy,
                                   wcskey=wcskey)[0]
                if len(wcs_xy) == 3:
                    # coords are wavelength, dec, ra
                    wcs_xy = np.array([wcs_xy[2], wcs_xy[1]])
                crpix_shifts[i] = -1 * (s - wcs_xy)
        result = crpix_shifts

    return np.array(result)


def expand_array(arr, shape, missing=np.nan):
    """
    Expands an array to a new shape

    New shape must be greater than the original in both dimensions.
    Original data will be inserted in the lower left corner (0, 0).
    The right and upper border will be filled with `missing`.

    Parameters
    ----------
    arr : numpy.ndarray
        array to expand (nrow, ncol)
    shape : array_like
        (y, x) dimensions of new shape
    missing : float, optional
        value with which to fill the right and upper borders after
        expansion.

    Returns
    -------
    numpy.ndarray
        expanded array (shape[0], shape[1])
    """
    if not isinstance(arr, np.ndarray):
        return
    if shape is None:
        return arr.copy()
    else:
        s0 = arr.shape
        result = np.full(shape, missing)
        result[0: s0[0], 0: s0[1]] = arr.copy()
    return result


def shift_set(dataset, offset, newshape=None, missing=np.nan):
    """
    Shifts an individual data set

    Note that the shift order is read via
    sofia_redux.instruments.forcast.getpar in
    sofia_redux.instruments.forcast.shift.

    Integration time is taken from the maximum normalization value.  This
    should not be necessary since it should have been calculated at
    `sofia_redux.instruments.forcast.merge`, but is done anyway.

    Parameters
    ----------
    dataset : tuple
        data : numpy.ndarray
        header : astropy.io.fits.header.Header
        variance : numpy.ndarray, optional
        normalization : numpy.ndarray, optional
    newshape : array_like
        (y, x) shape of any output arrays.  Must be greater than or
        equal to the original image shape in each dimension.  Original
        images are inserted in the bottom-left (0, 0) corner of the
        expanded image before shifting.
    offset : array_like
        (x, y) shift
    missing : float, optional
        value with which to fill the right and upper borders after
        expansion.

    Returns
    -------
    tuple
        shifted version of input dataset
    """
    data = expand_array(dataset[0], newshape, missing=missing)
    header = dataset[1].copy()
    hdinsert(header, 'COADX0', offset[0], 'X shift during coadd process')
    hdinsert(header, 'COADY0', offset[1], 'Y shift during coadd process')
    header['NAXIS1'] = data.shape[1]
    header['NAXIS2'] = data.shape[0]
    if len(dataset) > 2 and isinstance(dataset[2], np.ndarray):
        var = expand_array(dataset[2], newshape, missing=missing)
    else:
        var = None
    data, var = shift(data, offset, header=header,
                      variance=var, missing=missing)

    if len(dataset) > 3 and isinstance(dataset[3], np.ndarray):
        norm = expand_array(dataset[3], newshape, missing=missing)
        norm, _ = shift(norm, offset, missing=missing, order=0)
        hdinsert(header, 'EXPTIME', np.nanmax(norm), refkey='HISTORY')
    else:
        norm = None

    result = data, header
    if len(dataset) > 2:
        result = *result, var
    if len(dataset) > 3:
        result = *result, norm
    return result


def shift_datasets(datasets, shifts, refset, missing=np.nan):
    """
    Shifts datasets onto common frame

    Parameters
    ----------
    datasets : list of tuple
        tuple is of the form: data, header, variance, normalization
    shifts : numpy.ndarray
        (nimage, (x, y))
    refset : int
        Index of the dataset to use as the reference.
    missing : float, optional
        value with which to fill missing data points

    Returns
    -------
    list of tuples
    """
    if len(datasets) == 1:
        return [shift_set(datasets[0], shifts[0], missing=missing)]

    dout = []
    s = np.array(shifts).copy().astype(float)
    minvals = np.min(s, axis=0)
    s[:, minvals < 0] += 0 - np.floor(minvals[minvals < 0])

    # swap x and y in shifts to add to shapes
    shape_s = np.roll(s, 1, axis=1)
    shapes = np.array([tuple(d)[0].shape for d in datasets])

    maxshape = np.max(np.ceil(shape_s + shapes), axis=0).astype(int)

    for dset, offset in zip(datasets, s):
        dout.append(shift_set(dset, offset, newshape=maxshape,
                              missing=missing))

    # update all WCS to the reference set
    refwcs = WCS(dout[refset][1]).to_header(relax=True)
    for dset in dout:
        for key, value in refwcs.items():
            if key in dset[1]:
                dset[1][key] = value

    return dout


def resize_datasets(datasets, missing=np.nan):
    """
    Resize all datasets to the same shape

    Parameters
    ----------
    datasets

    Returns
    -------
    datasets
    """
    mx, my = 0, 0
    for dset in datasets:
        mx = dset[0].shape[1] if mx < dset[0].shape[1] else mx
        my = dset[0].shape[0] if my < dset[0].shape[0] else my
    maxshape = my, mx
    result = []
    for dset in datasets:
        newset = []
        if dset[0].shape != maxshape:
            newset.append(expand_array(dset[0], maxshape, missing=missing))
            header = dset[1].copy()
            header['NAXIS1'] = mx
            header['NAXIS2'] = my
            newset.append(dset[1].copy())
            if len(dset) > 2:
                newset.append(expand_array(dset[2], maxshape, missing=missing))
            if len(dset) > 3:
                newset.append(expand_array(dset[3], maxshape, missing=0))
            result.append(tuple(newset))
        else:
            result.append(dset)
    return result


def register_datasets(datasets, user_shifts=None, basehead=None, refset=0,
                      algorithm=None, initial_reference=None,
                      missing=np.nan):
    """
    Registers multiple sets of data to the same frame

    Parameters
    ----------
    datasets : array_like of tuples
        All elements are tuples.  Each tuple represents a data set consisting
        where the elements are as follows:

        1. numpy.ndarray:
           image array (nrow, ncol).  Image shapes are allowed to
           differ from others in the inputs data set.
        2. astropy.io.fits.header.Header:
           FITS header for the image array
        3. numpy.ndarray, optional:
           variance array (nrow, ncol).  Used to propagate variance.
           Must be the same shape as the data array in the first
           element.
        4. numpy.ndarray, optional:
           normalization map (nrow, ncol) to propagate.  Must be the
           same shape as the data array in the first element.

    refset : int, optional
        Index of the dataset to use as the reference.  Default is the first
        set in the list (0).
    user_shifts : array_like of array_like, optional
        User shifts if required.  No shift may be represented by None.
        All other shifts should be supplied as (dx, dy).  Length must be
        the same length as datasets
    basehead : astropy.io.fits.header.Header, optional
        FITS header to update with HISTORY messages
    algorithm : str, optional
        Registration and coadding algorithm.  If None, defaults to the
        coadding algorithm determined by the drip configuration or
        reference header.
    initial_reference : array_like, optional
        An optional reference to pass if an initial shift on the reference
        dataset is required.
    missing : float, optional
        Value with which to fill missing data points following a shift

    Returns
    -------
    list of tuple
        registered datasets
    """
    if not hasattr(datasets, '__len__'):
        log.error('Invalid datasets - nothing to coadd')
        return
    for i, dset in enumerate(datasets):
        if len(dset) < 2 or len(dset) > 4:
            log.error("dataset %i - invalid elements" % i)
            return
        elif not isinstance(dset[0], np.ndarray) or len(dset[0].shape) != 2:
            log.error("dataset %i - invalid image" % i)
            return
        elif not isinstance(dset[1], fits.header.Header):
            log.error("dataset %s - invalid header" % i)
            return
        if len(dset) > 2 and isinstance(dset[2], np.ndarray):
            if dset[2].shape != dset[0].shape:
                log.error("dataset %s - invalid variance" % i)
                return
        if len(dset) > 3 and isinstance(dset[3], np.ndarray):
            if dset[3].shape != dset[0].shape:
                log.error("dataset %s - invalid normmap" % i)
                return

    if user_shifts is not None:
        if not hasattr(user_shifts, '__len__') or \
                len(user_shifts) != len(datasets):
            log.error("invalid user offsets")
            return

    datasets = resize_datasets(datasets, missing=missing)

    with catch_warnings():
        simplefilter('ignore')
        shifts = get_shifts(datasets, user_shifts=user_shifts, refset=refset,
                            algorithm=algorithm,
                            initial_reference=initial_reference,
                            basehead=basehead)
    failure = False
    for idx, s in enumerate(shifts):
        if s is None:
            log.error("failed to register dataset %i" % idx)
            failure = True
    if failure:
        return

    registered = shift_datasets(datasets, shifts, refset,
                                missing=missing)

    return registered
