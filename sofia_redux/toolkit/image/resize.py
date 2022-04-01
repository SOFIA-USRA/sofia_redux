# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.lib import NumpyVersion
import scipy
from scipy import ndimage

from sofia_redux.toolkit.image.utilities import (
    to_ndimage_mode, clip_output)


__all__ = ['resize']


def resize(image, output_shape, order=None, mode='reflect', cval=0, clip=True,
           anti_aliasing=None, anti_aliasing_sigma=None):
    """
    Replacement for `skimage.resize`.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    output_shape : iterable
        Size of the generated output image `(rows, cols[, ...][, dim])`. If
        `dim` is not provided, the number of channels is preserved. In case the
        number of input channels does not equal the number of output channels a
        n-dimensional interpolation is applied.
    order : int, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5.
    mode : str, optional
        One of {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}. Points
        outside the boundaries of the input are filled according to the
        given mode.  Modes match the behaviour of `numpy.pad`.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    clip : bool, optional
        Whether to clip the output to the range of values of the input image.
        This is enabled by default, since higher order interpolation may
        produce values outside the given input range.
    anti_aliasing : bool, optional
        Whether to apply a Gaussian filter to smooth the image prior
        to downsampling. It is crucial to filter when downsampling
        the image to avoid aliasing artifacts. If not specified, it is set to
        True when downsampling an image whose data type is not bool.
    anti_aliasing_sigma : {float, tuple of floats}, optional
        Standard deviation for Gaussian filtering used when anti-aliasing.
        By default, this value is chosen as (s - 1) / 2 where s is the
        downsampling factor, where s > 1. For the up-size case, s < 1, no
        anti-aliasing is performed prior to rescaling.

    Returns
    -------
    resized : numpy.ndarray
        Resized version of the input.
    """
    input_shape = image.shape
    input_type = image.dtype

    if order is None:
        order = 0 if input_type == bool else 1

    if input_type == np.float16:
        image = image.astype(np.float32)

    if anti_aliasing is None:
        anti_aliasing = (not input_type == bool
                         and any(x < y for x, y
                                 in zip(output_shape, input_shape)))

    if input_type == bool and anti_aliasing:
        raise ValueError("anti_aliasing must be False for boolean images")

    factors = np.divide(input_shape, output_shape)
    image = image.astype(float)
    bounds = np.array([np.nanmin(image), np.nanmax(image)]) if clip else None
    ndi_mode = to_ndimage_mode(mode)

    if anti_aliasing:
        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
        else:
            anti_aliasing_sigma = \
                np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
            if np.any(anti_aliasing_sigma < 0):
                raise ValueError("Anti-aliasing standard deviation must be "
                                 "greater than or equal to zero")

        image = ndimage.gaussian_filter(image, anti_aliasing_sigma,
                                        cval=cval, mode=ndi_mode)

    if NumpyVersion(scipy.__version__) >= '1.6.0':  # pragma: no cover
        # The grid_mode kwarg was introduced in SciPy 1.6.0
        zoom_factors = [1 / f for f in factors]
        out = ndimage.zoom(image, zoom_factors, order=order, mode=ndi_mode,
                           cval=cval, grid_mode=True)
    else:  # pragma: no cover
        # Just do N-D interpolation
        coord_arrays = [factors[i] * (np.arange(d) + 0.5) - 0.5
                        for i, d in enumerate(output_shape)]

        coord_map = np.array(np.meshgrid(*coord_arrays,
                                         sparse=False,
                                         indexing='ij'))
        out = ndimage.map_coordinates(image, coord_map, order=order,
                                      mode=ndi_mode, cval=cval)
        clip_output(bounds, out, mode, cval, clip)
    return out
