# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy.stats import sigma_clip
import numpy as np


__all__ = ['combine_images']


def combine_images(data, variance=None, method='median', weighted=True,
                   robust=True, returned=True, **kwargs):
    """Combine input image arrays.

    Uses masked arrays to ignore input NaNs and to clip outliers if necessary.
    Masked pixels in the output are replaced with NaN before returning.

    Parameters
    ----------
    data : `list` of numpy.ndarray or numpy.ndarray
        Input data to combine.  Dimensions must match for all arrays.
        If ndarray is provided, images will be combined along axis 0.
    variance : `list` of numpy.ndarray or numpy.ndarray, optional
        Variance associated with input data.  List length and features
        must match `data`.
    method : {'mean', 'median', 'sum'}, optional
        Combination function to use.
    weighted : bool, optional
        If True and method is 'mean', the input variance will be used
        to weight the mean combination.  Ignored if variance is not
        provided.
    robust : bool, optional
        If True, the threshold and maxiters parameters will be used
        to reject outliers before combination.  Outlier rejection is
        performed via `astropy.stats.sigma_clip`.
    returned : bool, optional
        If True, return the combined variance in addition to the combined
        image.  Variance will be propagated if provided, or calculated from
        the RMS of each pixel over all images.
    **kwargs
        Additional parameters to pass to `astropy.stats.sigma_clip`
        when robust=True.  'axis' and 'masked' keywords are ignored
        if provided.

    Returns
    -------
    tuple of numpy.ndarray
        Return value is (data, variance) for the combined array.
        If input variance is provided, it is propagated; otherwise,
        the returned variance is calculated from the input data.

    Raises
    ------
    ValueError
        Any improper inputs.
    """

    try:
        nimage = len(data)
        dshape = data[0].shape
    except (TypeError, IndexError, AttributeError):
        raise ValueError('Input data is not array-like.')
    if nimage == 1:
        raise ValueError('Only one image provided; no data to combine.')

    for darray in data:
        if darray.shape != dshape:
            raise ValueError('Input data features do not match.')
    if variance is not None:
        if len(variance) != nimage:
            raise ValueError('Variance list does not match data list.')
        for varray in variance:
            if varray.shape != dshape:
                raise ValueError('Input variance features do not match.')
    else:
        weighted = False

    method = str(method).lower()
    if method not in ['mean', 'median', 'sum']:
        raise ValueError("Invalid combination method: '{}'.".format(method))

    masked_array = np.ma.MaskedArray(data, mask=np.isnan(data))
    if robust:
        if 'axis' in kwargs:
            del kwargs['axis']
        if 'masked' in kwargs:
            del kwargs['masked']
        masked_array = sigma_clip(masked_array, axis=0, masked=True, **kwargs)

    if variance is not None:
        masked_array.mask |= np.isnan(variance)
        masked_var = np.ma.masked_array(variance, mask=masked_array.mask)
    else:
        masked_var = None

    if method == 'sum':
        combined = np.ma.sum(masked_array, axis=0)
        if returned:
            if masked_var is not None:
                combined_var = np.ma.sum(masked_var, axis=0)
            else:
                combined_var = np.ma.var(masked_array, axis=0)
        else:
            combined_var = None
    elif method == 'mean':
        if weighted and masked_var is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                weights = 1 / masked_var
            combined, wtsum = np.ma.average(masked_array, axis=0,
                                            weights=weights, returned=True)
            combined_var = 1 / wtsum if returned else None
        else:
            combined = np.ma.mean(masked_array, axis=0)
            if returned:
                if masked_var is not None:
                    combined_var = (np.ma.sum(masked_var, axis=0)
                                    / np.ma.count(masked_var, axis=0)**2)
                else:
                    combined_var = np.ma.var(masked_array, axis=0)
            else:
                combined_var = None
    else:
        # method == 'median':
        combined = np.ma.median(masked_array, axis=0)
        if returned:
            if masked_var is not None:
                # variance is pi/2 * variance for mean
                combined_var = (np.ma.sum(masked_var, axis=0)
                                / np.ma.count(masked_var, axis=0)**2)
                combined_var *= np.pi / 2.0
            else:
                combined_var = np.ma.var(masked_array, axis=0)
        else:
            combined_var = None

    combined = np.ma.filled(combined, fill_value=np.nan)
    if combined_var is not None:
        combined_var = np.ma.filled(combined_var, fill_value=np.nan)

    return (combined, combined_var) if returned else combined
