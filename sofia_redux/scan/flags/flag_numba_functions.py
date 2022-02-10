# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

from sofia_redux.toolkit.splines import spline_utils

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['set_flags', 'unflag', 'flatten_nd_indices',
           'is_flagged', 'is_unflagged', 'get_mem_correction',
           'set_new_blank_value']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def set_flags(flag_array, flag, indices=None):  # pragma: no cover
    """
    Add a flag indicator to an array of flags.

    Parameters
    ----------
    flag_array : numpy.ndarray (int)
        An array of flag values to be updated in-place of shape (n,).
    flag : int
        The integer flag to set.
    indices : numpy.ndarray (int) or int, optional
        An array indicating which `flag_array` indices to flag with `flag`.
        The default is all indices.

    Returns
    -------
    None
    """
    flat_array = flag_array.flat
    if indices is None:
        for index in range(flag_array.size):
            flat_array[index] |= flag
    else:
        if flag_array.ndim > 1:
            flat_indices = flatten_nd_indices(indices, flag_array.shape).flat
        else:
            flat_indices = np.asarray(indices).flat

        for index in flat_indices:
            flat_array[index] |= flag


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def unflag(flag_array, flag=None, indices=None):  # pragma: no cover
    """
    Remove a flag indicator from an array of flags.

    Parameters
    ----------
    flag_array : numpy.ndarray (int)
        An array of flag values to be updated in-place of shape (n,).
    flag : int, optional
        The integer flag to remove.  If not supplied, all flags are removed
        (set to zero).
    indices : numpy.ndarray (int) or int, optional
        An array indicating which `flag_array` indices to flag with `flag`.
        The default is all indices.

    Returns
    -------
    None
    """
    flat_array = flag_array.flat
    if indices is None:
        if flag is None:
            for index in range(flag_array.size):
                flat_array[index] = 0
        else:
            for index in range(flag_array.size):
                flag_value = flat_array[index]
                if (flag_value & flag) == 0:
                    continue
                flat_array[index] = flag_value ^ flag
        return

    if flag_array.ndim > 1:
        flat_indices = flatten_nd_indices(indices, flag_array.shape).flat
    else:
        flat_indices = np.asarray(indices).flat
    if flag is None:
        for index in flat_indices:
            flat_array[index] = 0
    else:
        for index in flat_indices:
            flag_value = flat_array[index]
            if (flag_value & flag) == 0:
                continue
            flat_array[index] = flag_value ^ flag


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def flatten_nd_indices(indices, array_shape):  # pragma: no cover
    """
    Converts ND array indices to flat indices.

    Parameters
    ----------
    indices : tuple or numpy.ndarray (int)
        The ND indices in the form that would be returned by
        :func:`np.nonzero`.
    array_shape : tuple (int) or numpy.ndarray (int)
        The shape of the array for which to generate flat indices.

    Returns
    -------
    array.flat, flat_indices : numpy.ndarray, numpy.ndarray (int)
    """
    n_dimensions = len(indices)
    _, _, steps = spline_utils.flat_index_mapping(np.asarray(array_shape))
    n = indices[0].size
    flat_indices = np.zeros(n, dtype=nb.int64)
    for dimension in range(n_dimensions):
        step = steps[dimension]
        index_line = indices[dimension]
        for i in range(n):
            flat_indices[i] += step * index_line[i]
    return flat_indices


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def is_flagged(flag_array, flag=None, exact=False):  # pragma: no cover
    """
    Return whether a flag array is flagged with a given flag.

    Parameters
    ----------
    flag_array : numpy.ndarray (int)
        An array of integer flags.
    flag : int, optional
        The flag to check.  If not supplied, any non-zero flag will be
        returned as a `True` value.
    exact : bool, optional
        If `True`, a flagged result is one that exactly matches the flag.
        Otherwise, a flagged result is one which contains the flag.

    Returns
    -------
    flagged : numpy.ndarray (bool)
        A mask the same shape as `flag_array` where `True` indicates an
        element is flagged.
    """
    mask = np.empty(flag_array.shape, dtype=nb.b1)
    flat_mask = mask.flat
    flat_flag = flag_array.flat
    if flag is None:
        for i in range(flag_array.size):
            flat_mask[i] = flat_flag[i] != 0
    elif flag == 0:
        for i in range(flag_array.size):
            flat_mask[i] = flat_flag[i] == 0
    elif exact:
        for i in range(flag_array.size):
            flat_mask[i] = flat_flag[i] == flag
    else:
        for i in range(flag_array.size):
            flat_mask[i] = (flat_flag[i] & flag) != 0
    return mask


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=True)
def is_unflagged(flag_array, flag=None, exact=False):  # pragma: no cover
    """
    Return whether a flag array is unflagged by the given flag.

    Parameters
    ----------
    flag_array : numpy.ndarray (int)
        An array of integer flags.
    flag : int, optional
        The flag to check.  If not supplied, returns `True` if `flagged_array`
        is zero.
    exact : bool, optional
        If `True`, an unflagged result is one that does not exactly match the
        flag.  Otherwise, an unflagged result is one which does not contain
        the flag.

    Returns
    -------
    unflagged : numpy.ndarray (bool)
        A mask the same shape as `flag_array` where `True` indicates an
        element is unflagged.
    """
    mask = np.empty(flag_array.shape, dtype=nb.b1)
    flat_mask = mask.flat
    flat_flag = flag_array.flat
    if flag is None:
        for i in range(flag_array.size):
            flat_mask[i] = flat_flag[i] == 0
    elif flag == 0:
        for i in range(flag_array.size):
            flat_mask[i] = flat_flag[i] != 0
    elif exact:
        for i in range(flag_array.size):
            flat_mask[i] = flat_flag[i] != flag
    else:
        for i in range(flag_array.size):
            flat_mask[i] = (flat_flag[i] & flag) == 0
    return mask


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_mem_correction(data, noise, multiplier=0.1, valid=None, model=None
                       ):  # pragma: no cover
    """
    Determine the Maximum-Entropy-Method (MEM) correction.

    The MEM correction is given by:

    dx = sign(x) * n * multiplier * log(sqrt(x^2 + n^2) / sqrt(m^2 + n^2))

    Where x is a data value, n is the noise, and m is the model.  Any invalid
    values (NaN, zero-divisions, marked invalid etc.) will result in zero
    valued MEM correction (dx) on output.

    Parameters
    ----------
    data : numpy.ndarray (float)
        A data array of values with arbitrary shape (shape,).
    noise : numpy.ndarray (float)
        The noise array with shape (shape,).
    multiplier : float, optional
        The Lagrange multiplier.
    valid : numpy.ndarray (bool), optional
        A boolean mask of shape (shape,) where `False` excludes an element from
        the MEM correction.
    model : numpy.ndarray (float), optional
        An optional model of shape (shape,).  By default, all model values are
        zero.

    Returns
    -------
    mem_correction : numpy.ndarray (float)
        The MEM correction of shape (shape,).
    """
    flat_data = data.flat
    flat_noise = noise.flat
    if valid is None:
        do_valid = False
        flat_valid = np.empty(0, dtype=nb.b1).ravel()
    else:
        do_valid = True
        flat_valid = valid.ravel()
    if model is None:
        do_model = False
        flat_model = np.empty(0, dtype=nb.float64).ravel()
    else:
        do_model = True
        flat_model = model.ravel()

    mem_values = np.empty_like(data)
    flat_mem = mem_values.flat

    for i in range(data.size):
        if do_valid and not flat_valid[i]:
            flat_mem[i] = 0.0
            continue
        value = flat_data[i]
        if np.isnan(value):
            flat_mem[i] = 0.0
            continue
        if do_model:
            target = flat_model[i]
            if np.isnan(target):
                flat_mem[i] = 0.0
                continue
        else:
            target = 0.0
        sigma = flat_noise[i]
        if np.isnan(sigma):
            flat_mem[i] = 0.0
            continue

        d_target = np.hypot(target, sigma)
        if d_target == 0:
            flat_mem[i] = 0.0
            continue
        d_value = np.hypot(value, sigma)

        mem_value = multiplier * sigma * np.log(d_value / d_target)
        if value < 0:
            mem_value *= -1
        flat_mem[i] = mem_value

    return mem_values


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def set_new_blank_value(data, old_blank, new_blank):  # pragma: no cover
    """
    Set all old blanking levels in the data to the new blanking level.

    Parameters
    ----------
    data : numpy.ndarray
        An arbitrarily shaped data array of arbitrary type.
    old_blank : int or float
    new_blank : int or float or None

    Returns
    -------
    None
    """
    if old_blank == new_blank or old_blank is new_blank or new_blank is None:
        return
    data = data.ravel()
    for i in range(data.size):
        if data[i] == old_blank or data[i] is old_blank:
            data[i] = new_blank
