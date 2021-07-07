# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.utilities.func import slicer
import warnings

__all__ = ['scaleimgs']


def scaleimgs(images, variances=None, axis=0, nan=True):
    """
    Scales a set of images to the median flux level of all images

    Determines the median signal of each image in the cube M_i. Then
    it computes the median of these values M and then determines
    scale factors to scale M_i ti M.  It then applied these scale
    factors to the images in the cubes.

    Parameters
    ----------
    images : array_like of float (shape)
        Images of arbitrary shape
    variances : array_like of float (shape), optional
        Variances to update in parallel with images
    axis : int, optional
        The axis determining separate images.
    nan : bool, optional
        If True, ignore NaNs when determining median values.

    Returns
    -------
    scaled_images, [scaled_variances] : numpy.ndarray, [numpy.ndarray]
        Images and optionally variances (if variances were supplied)
        scaled to the same median level.  Both arrays are of float
        type and (shape).
    """
    images = np.array(images).astype(float)
    dovar = variances is not None
    if dovar:
        variances = np.array(variances).astype(float)
        if variances.shape != images.shape:
            log.error("Images and variances shape mismatch")
            return

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian if nan else np.median
        nimages = images.shape[axis]
        medimages = np.zeros(nimages)
        slices = []
        for imagei in range(nimages):
            imslice = slicer(images, axis, imagei, ind=True)
            slices.append(imslice)
            medimages[imagei] = median(images[imslice])

    scale = np.full(nimages, np.nan)
    nzi = medimages != 0
    scale[nzi] = median(medimages) / medimages[nzi]
    vscale = scale ** 2 if dovar else None

    for imagei, imslicer in enumerate(slices):
        images[imslicer] *= scale[imagei]
        if dovar:
            variances[imslicer] *= vscale[imagei]

    if dovar:
        return images, variances
    else:
        return images
