# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from scipy.ndimage import interpolation, affine_transform
from skimage.transform import resize

from sofia_redux.toolkit.image.fill import image_naninterp
from sofia_redux.toolkit.interpolate.interpolate import line_shift


__all__ = ['shift', 'rotate', 'frebin', 'image_shift', 'rotate90',
           'unrotate90', 'register_image', 'upsampled_dft']


def upsampled_dft(data, upsampled_region_size,
                  upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This code is intended to provide the same result as if the following
    operations were performed:

        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size, ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError("shape of upsampled region sizes must be equal "
                             "to input data's number of dimensions.")

    if axis_offsets is None:
        axis_offsets = [0, ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError("number of axis offsets must be equal to input "
                             "data's number of dimensions.")

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = ((np.arange(ups_size) - ax_offset)[:, None]
                  * np.fft.fftfreq(n_items, upsample_factor))
        kernel = np.exp(-im2pi * kernel)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def shift(data, offset, order=1, missing=np.nan, nan_interpolation=0.0,
          missing_limit=1e-3, mode='constant'):
    """
    Shift an image by the specified amount.

    Uses interpolation to do sub-pixel shifts if desired.  NaNs are
    safely handled.

    Parameters
    ----------
    data : numpy.ndarray
        The image array to shift_image (nrow, ncol)
    offset : array_like
        The shift along each axis.  If a float, shift is the same for
        all axis.  If an array, shift should contain one value for each
        axis.
    order : int, optional
        Interpolation order.  0 shifts by integer pixels; 1 shifts
        using bilinear interpolation; 2-5 shift_image using cubic-spline
        interpolation of the same order.
    missing : float or int, optional
        Value to fill past  edges of input if `edge` is 'constant'.
    nan_interpolation : float, optional
        NaN values must be replaced by a real value before shifting.
        Setting this value to None will result in NaN values being replaced
        by an interpolated value using the Clough-Tocher scheme.
    missing_limit : float, optional
        The fraction of NaNs that were excluded from the interpolation.
        If the interpolation exceeds this limit, the value will be
        replaced by `missing`.
    mode : str, optional
         Points outside the boundaries of the input are filled
         according to the given mode ('constant', 'nearest', 'reflect',
         or 'wrap').

    Returns
    -------
    numpy.ndarray
        shifted image image array (nrow, ncol)
    """
    if not isinstance(data, np.ndarray):
        log.error("data must be %s" % np.ndarray)
        return
    ndim = len(data.shape)
    if not hasattr(offset, '__len__'):
        offset = [offset] * ndim
    if len(offset) != ndim:
        log.error("invalid offset %s" % repr(offset))
        return

    try:
        order = int(order)
    except (TypeError, ValueError):
        log.error("invalid order %s" % repr(order))
        return
    if order < 0 or order > 5:
        log.error("order must be between in the range 0-5")
        return

    if order >= 1:
        if np.allclose(offset, np.floor(offset)):
            order = 0

    nans = np.isnan(data)
    cval = np.nan if missing is None else missing
    if not nans.any():
        return interpolation.shift(data, offset, order=order,
                                   cval=cval, mode=mode)

    if nan_interpolation is None:
        shifted = image_naninterp(data)
    else:
        if np.isnan(nan_interpolation):
            raise ValueError("nan_interpolation must be finite")
        shifted = data.copy()
        shifted[nans] = nan_interpolation

    shifted = interpolation.shift(shifted, offset, order=order,
                                  cval=cval, mode=mode)

    if missing is not None:
        nanmask = interpolation.shift(
            nans.astype(float), offset, order=order, cval=1, mode=mode)
        shifted[np.abs(nanmask) > missing_limit] = missing

    return shifted


def rotate(data, angle, order=1, missing=np.nan, nan_interpolation=0.0,
           missing_limit=1e-3, mode='constant', pivot=None):
    """
    Rotate an image.

    Rotates an image `angle` degrees clockwise around `center`.
    I cannot find any interpolation algorithms in Python that
    handle NaNs (quickly), so do it myself.

    Parameters
    ----------
    data : numpy.ndarray
        2 dimensional data array to rotate
    angle : float
        angle in degrees to rotate the image clockwise around
        center
    order : int, optional
        interpolation order.  0 rotates by integer pixels; 1 shifts
        using bilinear interpolation; 2-5 shift_image using
        cubic-spline interpolation of the same order.
    missing : float or int, optional
        value to fill past  edges of input if `edge` is 'constant'.
        Default is numpy.nan.
    nan_interpolation : float, optional
        NaN values must be replaced by a real value before rotating.
        Setting this value to None will result in NaN values being replaced
        by an interpolated value using the Clough-Tocher scheme.
    missing_limit : float, optional
        use as a threshold 0 to 1 to detect missing values
    mode : str, optional
        Points outside the boundaries of the nearest input are filled
        according to the given mode ('constant', 'nearest', 'reflect',
        or 'wrap').  Default is 'constant'.
    pivot : array_like
        (y, x) coordinate of the center of rotation

    Returns
    -------
    numpy.ndarray or None
       The rotated image or None on failure
    """
    if len(data.shape) != 2:
        log.error("data must be 2-dimensional")
        return

    cval = np.nan if missing is None else missing
    rads = np.radians(angle)
    cosa = np.cos(rads)
    sina = np.sin(rads)
    matrix = np.array([[cosa, sina], [-sina, cosa]], dtype=np.float64)
    center = np.array([data.shape[0] - 1, data.shape[1] - 1],
                      dtype=np.float64) / 2
    if pivot is None:
        offset = center - np.dot(matrix, center)
    else:
        piv = np.array(pivot, dtype=np.float64)
        if piv.shape != (2,):
            log.error("invalid pivot %s: will not rotate" % repr(pivot))
            return
        offset = piv - np.dot(matrix, piv)

    kwargs = {'order': order, 'cval': cval, 'prefilter': False,
              'mode': mode, 'output_shape': data.shape, 'offset': offset}

    nans = np.isnan(data)
    do_nans = nans.any()

    if angle % 360 == 0:
        clean_data = data.copy()
        if missing is not None:
            clean_data[nans] = missing
        return clean_data

    if do_nans:
        if nan_interpolation is None:
            clean_data = image_naninterp(data)
        else:
            if np.isnan(nan_interpolation):
                raise ValueError("nan_interpolation must be finite")
            clean_data = data.copy()
            clean_data[nans] = nan_interpolation
    else:
        clean_data = data

    output = affine_transform(clean_data, matrix, **kwargs)

    if not do_nans:
        return output.astype(data.dtype)

    kwargs['cval'] = 1.0

    if missing is not None:
        nanmask = affine_transform(nans.astype(float), matrix, **kwargs)
        output[np.abs(nanmask) > missing_limit] = missing

    return output.astype(data.dtype)


def frebin(data, shape, total=False, order=None,
           anti_aliasing=None, mode='reflect'):
    """
    Rebins an array to new shape

    Parameters
    ----------
    data : numpy.ndarray
        input data (nrow, ncol)
    shape : 2-tuple of int
        output data shape (nrow2, ncol2)
    total : bool, optional
        if True the sum of the output data will equal the sum
        of the input data.
    order : int, optional
        0 = nearest neighbor
        1 = bi-linear
        2 = bi-quadratic
        3 = bi-cubic
        4 = bi-quartic
        5 = bi-quintic
        By default, nearest neighbor will be used if the output
        shape is an integer factor of the input data shape.
        Otherwise, bi-linear interpolation will be used.
    anti_aliasing : bool, optional
        passed into `resize`
    mode : str, optional
        passed into `resize`

    Returns
    -------
    numpy.ndarray
       resized data array (nrow2, ncol2)
    """
    if data.shape == shape:
        return data.copy()

    if order is None:
        for s1, s2 in zip(data.shape, shape):
            if (s1 % s2 != 0) and (s2 % s1 != 0):
                order = 1  # linear
                break
        else:
            order = 0  # nearest neighbor

    if anti_aliasing is None:
        for s1, s2 in zip(data.shape, shape):
            if s2 > s1:
                anti_aliasing = True
                break
        else:
            anti_aliasing = False

    if np.isnan(data).any():
        d = data.copy()
        select = np.isnan(d)
        mask = np.full_like(d, 1)
        mask[select] = 0
        d[select] = 0
        rebinned = resize(d, shape, order=order,
                          mode=mode, anti_aliasing=anti_aliasing)
        mask = resize(mask, shape, order=order,
                      mode=mode, anti_aliasing=anti_aliasing)
        nzi = mask >= 0.5
        rebinned[nzi] /= mask[nzi]
        rebinned[~nzi] = np.nan
    else:
        rebinned = resize(data, shape, order=order,
                          mode=mode, anti_aliasing=anti_aliasing)
    if total:
        fac = np.product(data.shape) / np.product(shape)
        rebinned *= fac

    return rebinned


def image_shift(data, shifts, order=3, missing=np.nan):
    """
    Shifts an image by x and y offsets

    Uses `line_shift` to shift the entire image in the x-direction
    followed by a shift in the y-direction.

    Parameters
    ----------
    data : array_like
        2 dimensional (y, x) array to shift_image
    shifts : Sequence of (int or float)
        [xoffset, yoffset] (additive)
    order : int
        0 : integer shifts
        1 : linear interpolation
        2-5: spline order of interpolation.  3=cubic.
    missing : int or float
        Edges are treated as hard limits.  Values outside of these limits
        will be replaced with the missing value defined here.

    Returns
    -------
    np.ndarray
        A shifted copy of the input data
    """
    result = np.empty_like(data)
    result.fill(missing)
    for yind in np.arange(data.shape[0]):
        result[yind, :] = line_shift(
            data[yind, :], shifts[0], missing=missing, order=order)
    for xind in np.arange(data.shape[1]):
        result[:, xind] = line_shift(
            result[:, xind], shifts[1], order=order, missing=missing)
    return result


def rotate90(image, direction):
    """
    Replicates IDL rotate function

    In the table below, (X0, Y0) indicates the original subscripts, and
    (X1, Y1) are the subscripts of the resulting array.  The notation
    Y0 indicates a reversal of the Y axis, Y1 = Ny - Y0 - 1.

    ========= ========== ============= ==== ====
    Direction Transpose? AntiClockwise  X1   Y1
    ========= ========== ============= ==== ====
    0         N            0             X0   Y0
    1         N           90            -Y0   X0
    2         N          180            -X0  -Y0
    3         N          270             Y0  -X0
    4         Y            0             Y0   X0
    5         Y           90            -X0   Y0
    6         Y          180            -Y0  -X0
    7         Y          270             X0  -Y0
    ========= ========== ============= ==== ====

    Parameters
    ----------
    image : array_like
        (N, M) 2 dimensional array to rotate.
    direction : int
        See above for rotate value effect.  Direction is taken as
        modulo 8.

    Returns
    -------
    numpy.ndarray
    """
    image = np.asarray(image)
    ndim = image.ndim
    if ndim not in [1, 2]:
        raise ValueError("Only 1 or 2 features allowed")
    if ndim == 1:
        image = image[:, None]
    shape1 = image.shape

    rot = int(direction) % 8
    if rot >= 4:
        image = image.T

    image = np.rot90(image, k=(rot % 4), axes=(1, 0))
    if image.shape == shape1 and ndim == 1:
        image = image[:, 0]

    return image


def unrotate90(image, direction):
    """
    Un-rotates an image using IDL style rotation types

    The table below lists the different IDL rotation directions
    (The rotation applied to the image that you now wish to remove).
    (X0, Y0) indicates the original subscripts, and (X1, Y1) are
    the subscripts of the resulting array.  The notation Y0
    indicates a reversal of the Y axis, Y1 = Ny - Y0 - 1.

    ========= ========== ============= ==== ====
    Direction Transpose? AntiClockwise  X1   Y1
    ========= ========== ============= ==== ====
    0         N            0             X0   Y0
    1         N           90            -Y0   X0
    2         N          180            -X0  -Y0
    3         N          270             Y0  -X0
    4         Y            0             Y0   X0
    5         Y           90            -X0   Y0
    6         Y          180            -Y0  -X0
    7         Y          270             X0  -Y0
    ========= ========== ============= ==== ====

    This function applies the opposite rotation to the input
    image.

    Parameters
    ----------
    image : array_like of float
        (nrow, ncol) array to rotate.
    direction : int
        The IDL rotation type to "unrotate" the image by.

    Returns
    -------
    numpy.ndarray
        (nrow, ncol) or (ncol, nrow) depending on rotation type.
    """
    # Yes, I know there are lots of clever ways to do this in 1-2
    # lines, but it's displayed like this for clarity.
    rot = int(direction) % 8
    if rot == 0:
        return rotate90(image, 0)
    elif rot == 1:
        return rotate90(image, 3)
    elif rot == 2:
        return rotate90(image, 2)
    elif rot == 3:
        return rotate90(image, 1)
    elif rot == 4:
        return rotate90(image, 4)
    elif rot == 5:
        return rotate90(rotate90(image, 3), 4)
    elif rot == 6:
        return rotate90(rotate90(image, 2), 4)
    elif rot == 7:
        return rotate90(rotate90(image, 1), 4)
    else:
        raise ValueError("should not be able to get here")  # pragma: no cover


def register_image(image, reference, upsample=1,
                   maxshift=None, shift0=None):
    """
    Return the pixel offset between an image and a reference

    Uses cross-correlation (phase correlation) to find the pixel
    shift between an image and a reference image.  If `upsample`
    is greater than one, accuracy is increased to a subpixel
    level by upsampling a DFT in the neighborhood of the
    integer pixel solution.

    Parameters
    ----------
    image : numpy.ndarray
        Image to register (nrow, ncol).  Must be the same shape
        as `reference`.
    reference : numpy.ndarray
        Reference image (nrow, ncol).  Must be the same shape as
        `image`.
    upsample : int, optional
        Images will be registered to within `1 / upsample`.  For
        example, upsample = 100 results in registration to within
        1/100th of a pixel.
    maxshift : int or float or array_like, optional
        The maximum shift allowed in each dimension.  There may
        be more than one solution when images contain repeating
        patterns.  Therefore, set maxshift to search for maximum
        correlation within `init - maxshift` to `init + maxshift`.
        An array maxshift should have the same length as the image
        features whereas a single valued maxshift will be applied
        to all features.  None results in a search over the full
        image domain.  Order of features is (x, y, z, etc.)
    shift0 : array_like
        An initial registration estimate in each dimension.  This
        is useful when repeating patterns are present in the image
        and you wish to center the search at a specific location.
        Best used in conjunction with `maxshift`.  None centers the
        search around the center of the reference image.  Order of
        features is (x, y, z, etc.)

    Returns
    -------
    numpy.array
        Pixel registration in the (x, y, z, etc.) order.  This is the
        pixel shift of `reference` relative to `image` i.e. apply this
        shift to image to map it onto reference.
    """
    ndim = len(image.shape)
    if maxshift is not None:
        if hasattr(maxshift, '__len__'):
            maxshift = np.array(maxshift)
        else:
            maxshift = np.array([maxshift])
        if len(maxshift) == 1 and ndim > 1:
            maxshift = np.full(ndim, maxshift[0])
    if maxshift is not None and len(maxshift) != ndim:
        raise ValueError("Invalid maxshift length %s" % repr(maxshift))

    if shift0 is not None:
        if hasattr(shift0, '__len__'):
            shift0 = np.array(shift0)
        else:
            shift0 = np.array([shift0] * ndim)
    else:
        shift0 = np.zeros(len(image.shape))
    if len(shift0) != ndim:
        raise ValueError("Invalid shift0 length %s" % repr(shift0))
    if image.shape != reference.shape:
        raise ValueError("Image shape does not match reference shape")

    # apply initial offset now if supplied - integer only
    img = image.copy()
    img[np.isnan(img)] = np.nanmedian(img)
    ref = reference.copy()
    ref[np.isnan(reference)] = np.nanmedian(reference)

    ishift = np.array(list(np.fix(x) for x in shift0))
    if not (ishift == 0).all():
        img = shift(img, np.flip(ishift), order=0, mode='wrap')
    source = np.array(img, dtype=np.complex128, copy=False)
    sfreq = np.fft.fftn(source)
    rfreq = np.fft.fftn(ref)

    # calculate integer pixel shift
    product = sfreq * rfreq.conj()
    cc_image = np.fft.ifftn(product)
    shape = cc_image.shape

    if maxshift is not None:
        maxarr = np.zeros_like(cc_image)
        grids = np.meshgrid(*(np.arange(x) for x in np.flip(shape)))
        select = np.full(shape, True)
        for d, grid, mshift in zip(np.flip(shape), grids, maxshift):
            select &= (grid < mshift) | (grid > (d - mshift))
        maxarr[select] = cc_image[select]
    else:
        maxarr = cc_image.copy()

    maxima = np.unravel_index(np.argmax(maxarr.real), shape)
    mid = np.array([np.fix(dim / 2) for dim in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > mid] -= np.array(shape)[shifts > mid]

    if upsample > 1:
        # Initial shift estimate in upsampled grid
        # Taken from skimage.feature.register_translation
        shifts = np.round(shifts * upsample) / upsample
        upsampled_region_size = np.ceil(upsample * 1.5)
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample, dtype=np.float64)
        normalization = (sfreq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cc_image = upsampled_dft(
            product.conj(), upsampled_region_size,
            upsample, sample_region_offset).conj()  # upsample_factor
        cc_image /= normalization
        maxima = np.array(
            np.unravel_index(np.argmax(np.abs(cc_image)), cc_image.shape),
            dtype=np.float64)
        if (maxima == 0).all():  # pragma: no cover
            # we hit a boundary
            shifts.fill(np.nan)
        else:
            maxima -= dftshift
            shifts = shifts + maxima / upsample_factor

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(sfreq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    # if we shifted earlier, add it back on and return in x,y format
    result = ishift - np.flip(shifts)
    dtype = int if upsample == 1 else float
    return result.astype(dtype)
