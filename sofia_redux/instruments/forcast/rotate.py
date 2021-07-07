# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
import numpy as np
import skimage.transform as tf

from sofia_redux.toolkit.utilities.fits import add_history_wrap

addhist = add_history_wrap('Rotate')

__all__ = ['rotate']


def rotate(data, angle, header=None, variance=None, order=1, center=None,
           missing=np.nan, missing_limit=1e-3, strip_border=True):
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

    rows, cols = data.shape
    array_center = np.array([cols, rows]) / 2
    if center is not None:
        center = np.asarray(center)
    else:
        center = array_center

    # setup transforms
    # (below is from from tf.rotate; repeated here
    # in order to transform coordinates as well)
    tform1 = tf.SimilarityTransform(translation=center)
    tform2 = tf.SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = tf.SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1

    if strip_border:
        # determine shape of output image
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.ceil((out_rows, out_cols))

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = tf.SimilarityTransform(translation=translation)
        tform = tform4 + tform
    else:
        output_shape = None

    def point_rotate(x, y, do_header=False):
        if do_header:
            hpix = 1
        else:
            hpix = 0
        xy = np.array([x, y]) - hpix
        rotated = tform.inverse(xy)
        return rotated[0] + hpix

    def imgrot(img, use_order):
        # an odd quirk occasionally encountered: if data is not in
        # native byte order, the warp will fail.  Convert if necessary.
        if img.dtype.byteorder not in ('=', '|'):  # pragma: no cover
            img = img.byteswap().newbyteorder()
        return tf.warp(img, tform, output_shape=output_shape, order=use_order,
                       mode='constant', cval=0.0, clip=True,
                       preserve_range=False)

    def rotate_with_mask(img, use_order):
        if img is None:
            return
        nans = np.isnan(img)
        rimg = img.copy()
        rimg[nans] = 0
        mask = (~nans).astype(float)
        rimg = imgrot(rimg, use_order)
        rmask = imgrot(mask, use_order)
        nzi = np.abs(rmask) >= missing_limit
        rimg[nzi] /= rmask[nzi]
        rimg[~nzi] = np.nan
        rimg[np.isnan(rimg)] = missing
        return rimg

    rot = rotate_with_mask(data, order)
    var = rotate_with_mask(var, 0)

    # Strip NaN border if desired
    if strip_border:
        badmask = np.isnan(rot)
        badmask[rot == missing] = True
        strip_rows = np.all(badmask, axis=1)
        strip_cols = np.all(badmask, axis=0)
        xl, xu = np.argmax(~strip_cols), np.argmax(np.flip(~strip_cols))
        yl, yu = np.argmax(~strip_rows), np.argmax(np.flip(~strip_rows))
        xu, yu = len(strip_cols) - xu, len(strip_rows) - yu
        rot = rot[yl: yu, xl: xu]
        var = var[yl: yu, xl: xu] if dovar else None
        offset = np.array([xl, yl])

        # translate various coordinates - update WCS
        if 'CRPIX1' in header and 'CRPIX2' in header:
            new_pos = point_rotate(header['CRPIX1'], header['CRPIX2'],
                                   do_header=True)
            header['CRPIX1'] = new_pos[0] - offset[0]
            header['CRPIX2'] = new_pos[1] - offset[1]
        if 'SRCPOSX' in header and 'SRCPOSY' in header:
            new_pos = point_rotate(header['SRCPOSX'], header['SRCPOSY'])
            header['SRCPOSX'] = new_pos[0] - offset[0]
            header['SRCPOSY'] = new_pos[1] - offset[1]
        header['NAXIS1'] = rot.shape[1]
        header['NAXIS2'] = rot.shape[0]

        addhist(header, 'Resized image to (%s, %s) from (%s, %s)' %
                (rot.shape[1], rot.shape[0],
                 data.shape[1], data.shape[0]))

    return rot, var
