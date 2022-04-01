# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import add_history_wrap
from sofia_redux.toolkit.image.utilities import map_coordinates


addhist = add_history_wrap('Rotate')

__all__ = ['rotate', 'rotate_coordinates_about', 'rotate_point',
           'rotate_image', 'rotate_image_with_mask']


def rotate_coordinates_about(coordinates, center, angle,
                             shift=None, inverse=False):
    """
    Rotate the coordinates about a center point by a given angle.

    The forward transform is given by::

        rotated = rotate(coordinates + shift - center, angle) + center

    The inverse transform is given by::

        coordinates = rotate(rotated - center, -angle) + center - shift

    Parameters
    ----------
    coordinates : numpy.ndarray
        The coordinates to rotate of shape (2, shape,) where coordinates[0]
        contains the y-coordinates, and coordinates[1] contains the
        x-coordinates (numpy convention).
    center : numpy.ndarray
        The coordinate about which to perform the rotation of shape (2,) in
        (y, x) order (numpy convention).
    angle : int or float
        The angle in degrees by which to rotate the coordinates.
    shift : numpy.ndarray, optional
        An optional shift to apply to the coordinates of shape (
    inverse : bool, optional
        If `True`, perform the inverse rotation (rotate by `-angle`).

    Returns
    -------
    rotated_coordinates : numpy.ndarray (float)
        The rotated coordinates of shape (2, shape,).
    """
    radians = np.deg2rad(angle)
    do_shift = shift is not None
    if inverse:
        radians = -radians
    cos_a, sin_a = np.cos(radians), np.sin(radians)
    y, x = coordinates.astype(float)
    if do_shift and not inverse:
        y += shift[0]
        x += shift[1]
    y -= center[0]
    x -= center[1]
    xr = (cos_a * x) - (sin_a * y)
    yr = (sin_a * x) + (cos_a * y)
    y = yr + center[0]
    x = xr + center[1]
    if do_shift and inverse:
        y -= shift[0]
        x -= shift[1]
    return np.stack([y, x])


def rotate_point(y, x, center, angle, shift=None, for_header=False,
                 inverse=False):
    """
    Rotate a single (y, x) point about a given center.

    Parameters
    ----------
    y : int or float
        The y-coordinate.
    x : int or float
        The x-coordinate.
    center : numpy.ndarray
        The center of rotation of shape (2,) in (y, x) numpy ordering.
    angle : int or float
        The rotation angle in degrees.
    shift : numpy.ndarray, optional
        An optional shift to apply to the rotation prior to the rotation
        operation.  Should be of shape (2,) using the (y, x) numpy convention.
    for_header : bool, optional
        If `True`, indicates that (x, y) are taken from a FITS header (ordered
        from 1 rather than zero).
    inverse : bool, optional
        If `True`, perform the inverse rotation (rotate by `-angle`).

    Returns
    -------
    ry, rx : float, float
        The rotated x and y coordinates.
    """
    if for_header:
        offset = 1
    else:
        offset = 0

    yx = np.array([y, x]) - offset
    rotated = rotate_coordinates_about(yx, center, angle, shift=shift,
                                       inverse=inverse)
    rotated += offset
    return rotated[0], rotated[1]


def rotate_image(image, angle, center=None, order=1, shift=None,
                 output_shape=None, cval=np.nan, mode='constant', clip=True,
                 inverse=False, threshold=0.5):
    """
    Rotate an image about a point by a given angle.

    Rotation occurs using the same logic as :func:`rotate_coordinates_about`
    followed by an interpolation of the original image onto the rotated
    coordinates via splines of the supplied `order`.

    Parameters
    ----------
    image : numpy.ndarray
        The image to rotate with n_dimensions and of shape (shape,).
    angle : int or float
        The angle to rotate the image about `center` in degrees.
    center : numpy.ndarray, optional
        The coordinate for the center of rotation of shape (n_dimensions,)
        using the Numpy (y, x) convention.  The default is (image.shape-1)/2.
    order : int, optional
        The spline interpolation order.  Must be in the range 0-5.
    shift : numpy.ndarray, optional
        An optional shift to apply prior to the forward transform rotation of
        shape (n_dimensions,) using the Numpy (y, x) convention.  Please see
        :func:`rotate_coordinates_about` for further details.
    output_shape : tuple (int)
        The output shape for the rotated image of length n_dimensions using
        the Numpy (y, x) ordering convention.
    cval : int or float, optional
        Used in conjunction with `mode`='constant' to fill in values outside
        the boundaries of `image`.
    mode : str, optional
        Can take values of {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
        for which points outside the boundaries of the input are filled
        according to the given mode.  Modes match the behaviour of
        :func:`np.pad`.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    inverse : bool, optional
        If `True`, perform the inverse rotation (rotate by `-angle`).  Please
        see :func:`rotate_coordinates_about` for further details.
    threshold : float, optional
        Used in conjunction with `cval`=NaN and `mode`='constant'.  Should
        generally take values in the range -1 to 1 with a default of 0.5.
        This is used to better apply NaN `cval` boundaries as expected.  Points
        inside the boundaries are mapped to 1, and values outside are mapped to
        -1.  Points which map to values >= `threshold` are considered valid,
        while others will be set to NaN in the output.  Please see
        :func:`map_coordinates` for further details.

    Returns
    -------
    rotated : numpy.ndarray
        The rotated image with the same shape as `image` or `output_shape`.
    """
    image = image.astype(float)
    # an odd quirk occasionally encountered: if data is not in
    # native byte order, the warp will fail.  Convert if necessary.
    if image.dtype.byteorder not in ('=', '|'):  # pragma: no cover
        image = image.byteswap().newbyteorder()
    input_shape = np.array(image.shape)
    if output_shape is None:
        output_shape = input_shape
    if center is None:
        center = (np.asarray(input_shape) - 1) / 2

    coordinates = np.empty((2,) + tuple(output_shape), dtype=float)
    # In (y, x) order
    indices = np.indices(output_shape, dtype=float).reshape(2, -1)
    rotated = rotate_coordinates_about(indices, center, angle,
                                       shift=shift, inverse=inverse)
    for i in range(2):
        coordinates[i].flat = rotated[i]

    warped = map_coordinates(
        image, coordinates, order=order, mode=mode, cval=cval, clip=clip,
        threshold=threshold)

    return warped


def rotate_image_with_mask(image, angle, center=None, order=1, shift=None,
                           output_shape=None, inverse=False, threshold=0.5,
                           cval=np.nan, missing_limit=1e-3):
    """
    Rotate an image, using a mask for interpolation and edge corrections.

    Parameters
    ----------
    image : numpy.ndarray or None
        The image to rotate.  If `None` is supplied, it is also returned.
    angle : int or float
        The angle by which to rotate the image in degrees.
    center : numpy.ndarray, optional
        The point about which to rotate the image of shape (n_dimensions,)
        using the Numpy (y, x) dimensional ordering convention.  By default
        this is (image.shape-1)/2.
    order : int, optional
        The spline interpolation order.  Must be in the range 0-5.
    shift : numpy.ndarray, optional
        An optional shift to apply prior to the forward transform rotation of
        shape (n_dimensions,) using the Numpy (y, x) convention.  Please see
        :func:`rotate_coordinates_about` for further details.
    output_shape : tuple (int)
        The output shape for the rotated image of length n_dimensions using
        the Numpy (y, x) ordering convention.
    inverse : bool, optional
        If `True`, perform the inverse rotation (rotate by `-angle`).  Please
        see :func:`rotate_coordinates_about` for further details.
    threshold : float, optional
        Used in conjunction with `cval`=NaN and `mode`='constant'.  Should
        generally take values in the range -1 to 1 with a default of 0.5.
        This is used to better apply NaN `cval` boundaries as expected.  Points
        inside the boundaries are mapped to 1, and values outside are mapped to
        -1.  Points which map to values >= `threshold` are considered valid,
        while others will be set to NaN in the output.  Please see
        :func:`map_coordinates` for further details.
    cval : int or float, optional
        Used in conjunction with `mode`='constant' to fill in values outside
        the boundaries of `image`, and also to replace rotated NaN values or
        values that fall below `missing_limit` in the rotated mask.
    missing_limit : float, optional
        data weighted less than this fraction will be replaced with `cval`.

    Returns
    -------
    rotated : numpy.ndarray
        The rotated image with the same shape as `image` or `output_shape`.
    """
    if image is None:
        return image
    nans = np.isnan(image)
    data = image.copy()
    data[nans] = 0.0
    mask = np.logical_not(nans).astype(float)
    clip = not mask.all()
    kwargs = dict(center=center, order=order, shift=shift,
                  output_shape=output_shape, mode='constant', cval=0.0,
                  clip=clip, inverse=inverse, threshold=threshold)
    rotated = rotate_image(data, angle, **kwargs)
    rotated_mask = rotate_image(mask, angle, **kwargs)
    nzi = np.abs(rotated_mask) > missing_limit
    rotated[nzi] /= rotated_mask[nzi]
    rotated[~nzi] = cval
    rotated[np.isnan(rotated)] = cval
    return rotated


def rotate(data, angle, header=None, variance=None, order=1, center=None,
           missing=np.nan, missing_limit=1e-3, threshold=0.5,
           strip_border=True):
    """
    Rotate an image by the specified amount

    Rotates an image `angle` degrees clockwise around center.  A secondary
    image (variance) array may be supplied to rotate in parallel.

    If center is provided, it will be used as the center of rotation.
    Otherwise, the center is defined as array_shape / 2 with indexing
    starting at zero.  Note that WCS header uses indexing starting at 1.

    Parameters
    ----------
    data : numpy.ndarray
        The image array to shift (nrow, ncol)
    angle : float
        angle in degrees to rotate the image clockwise around
        center
    header : astropy.io.fits.header.Header
        If a header is provided, valid WCS will be updated
    variance :  numpy.ndarray, optional
        Variance array (nrow, ncol) to update in parallel with data
    order : int, optional
        Interpolation order.
            0 - nearest-neighbor
            1 - bilinear
            >=2 - spline of the same order
    center : array_like, optional
        If provided, will be used as the center of the rotation.
    missing : float, optional
        missing data fill value
    missing_limit : float, optional
        data weighted less than this fraction will be replaced with
        `missing`
    threshold : float, optional
        Used in conjunction with `cval`=NaN and `mode`='constant'.  Should
        generally take values in the range -1 to 1 with a default of 0.5.
        This is used to better apply NaN `cval` boundaries as expected.  Points
        inside the boundaries are mapped to 1, and values outside are mapped to
        -1.  Points which map to values >= `threshold` are considered valid,
        while others will be set to NaN in the output.  Please see
        :func:`map_coordinates` for further details.
    strip_border : bool, optional
        If True, strip any NaN only rows or columns from the edges
        of the rotated image.  WCS will be upated accordingly.  If
        False, the image will not be resized; data falling outside
        the image borders will be lost.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        The rotated image (nrow, ncol)
        The rotated variance (nrow, ncol) or None if not supplied
    """
    if not isinstance(header, fits.header.Header):
        header = fits.header.Header()
    var = variance.copy() if isinstance(variance, np.ndarray) else None
    if not isinstance(data, np.ndarray) or len(data.shape) != 2:
        log.error("invalid data")
        return

    dovar = isinstance(var, np.ndarray) and var.shape == data.shape
    var = None if not dovar else var.copy()
    if variance is not None and not dovar:
        log.warning("Variance not propagated (invalid variance)")

    if angle == 0:
        return data.copy(), var

    ny, nx = data.shape
    array_center = np.array([ny, nx]) / 2
    if center is None:
        center = array_center.copy()
    else:
        center = np.asarray(center)[::-1]  # FITS (x, y) to numpy (y, x)

    if strip_border:
        corners = np.asarray(
            [[0, 0, ny - 1, ny - 1],
             [0, nx - 1, 0, nx - 1]])
        rotated_corners = rotate_coordinates_about(
            corners, center, angle, inverse=True)
        min_y, max_y = np.min(rotated_corners[0]), np.max(rotated_corners[0])
        min_x, max_x = np.min(rotated_corners[1]), np.max(rotated_corners[1])
        ny2 = max_y - min_y + 1
        nx2 = max_x - min_x + 1
        output_shape = np.ceil((ny2, nx2)).astype(int)
        add_shift = np.array([min_y, min_x])
    else:
        output_shape = None
        add_shift = None

    kwargs = dict(center=center, shift=add_shift, output_shape=output_shape,
                  threshold=threshold, cval=missing,
                  missing_limit=missing_limit)

    rotated = rotate_image_with_mask(data, angle, order=order, **kwargs)
    var_rotated = rotate_image_with_mask(var, angle, order=0, **kwargs)

    # Strip NaN border if desired
    if strip_border:
        badmask = np.isnan(rotated)
        badmask[rotated == missing] = True
        strip_rows = np.all(badmask, axis=1)
        strip_cols = np.all(badmask, axis=0)
        xl, xu = np.argmax(~strip_cols), np.argmax(np.flip(~strip_cols))
        yl, yu = np.argmax(~strip_rows), np.argmax(np.flip(~strip_rows))
        xu, yu = len(strip_cols) - xu, len(strip_rows) - yu
        offset = np.array([yl, xl])
        rotated = rotated[yl: yu, xl: xu]
        var_rotated = var_rotated[yl: yu, xl: xu] if dovar else None

        # translate various coordinates - update WCS
        if 'CRPIX1' in header and 'CRPIX2' in header:
            px, py = header['CRPIX1'], header['CRPIX2']
            rpy, rpx = rotate_point(py, px, center, angle, shift=add_shift,
                                    for_header=True, inverse=True)
            rpy -= offset[0]
            rpx -= offset[1]
            header['CRPIX1'] = rpx
            header['CRPIX2'] = rpy

        if 'SRCPOSX' in header and 'SRCPOSY' in header:
            sx, sy = header['SRCPOSX'], header['SRCPOSY']
            spy, spx = rotate_point(sy, sx, center, angle, shift=add_shift,
                                    for_header=False, inverse=True)
            spy -= offset[0]
            spx -= offset[1]
            header['SRCPOSX'] = spx
            header['SRCPOSY'] = spy

        ny2, nx2 = rotated.shape
        header['NAXIS1'] = nx2
        header['NAXIS2'] = ny2

        addhist(header, f'Resized image to ({nx2}, {ny2}) from ({nx}, {ny})')

    return rotated, var_rotated
