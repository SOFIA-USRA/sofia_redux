# Licensed under a 3-clause BSD style license - see LICENSE.rst

from collections.abc import Mapping
from datetime import datetime
import gc
import math
import os
import re
import sys
import time

from astropy import log
from astropy.stats import gaussian_fwhm_to_sigma
import numpy as np
from numpy.lib.nanfunctions import _replace_nan, _copyto
from sofia_redux import toolkit as toolkit_module

__all__ = ['robust_bool', 'valid_num', 'natural_sort', 'goodfile',
           'date2seconds', 'str_to_value', 'slicer',
           'setnumber', 'gaussian_model', 'to_array_shape',
           'recursive_dict_update', 'stack', 'faststack',
           'taylor', 'bytes_string', 'remove_sample_nans', 'bitset',
           'julia_fractal', 'nantrim', 'nansum']


def robust_bool(value):
    """
    Check for 'truthy' values.

    Returns True if the case-insensitive string representation
    is 'y', '1', 'true', or 'yes; False otherwise.

    Parameters
    ----------
    value : str, bool, or int
        The value to test

    Returns
    -------
    bool
       True if `value` appears to indicate "True".  False otherwise.
    """
    return str(value).lower().strip() in ['y', '1', 'true', 'yes']


def valid_num(value):
    """
    Check for valid numbers.

    Returns True if the value can be cast to a float; False otherwise.

    Parameters
    ----------
    value : str, float, int, bool
        The value to test

    Returns
    -------
    bool
        True if `value` can be converted to a float, False otherwise.
    """
    try:
        result = float(value)
    except (ValueError, TypeError, AttributeError):
        result = None
    return isinstance(result, float)


def natural_sort(string_list, reverse=False):
    """Returns list sorted in a human friendly manner

    Typical python sorting will sort as 1,10,11,.. 2,20,21 when
    sorting a list of strings.  natural_sort will return the result
    1,2,3...10,11,12,...,19,20,21,...

    Parameters
    ----------
    string_list : `list` of str
        List of strings to sort
    reverse : bool, optional
        Reverse the sorting order

    Returns
    -------
    list of str
        Human sorted list of strings
    """
    def natural_sort_lambda(text):
        def int_or_string(text1):
            return int(text1) if text1.isdigit() else text1
        return [int_or_string(x) for x in re.split(r'(\d+)', text)]

    result = string_list[:]
    result.sort(key=natural_sort_lambda, reverse=reverse)
    return result


def goodfile(filename, read=True, write=False, execute=False, verbose=False):
    """
    Check if a file exists, and optionally if it has the correct permissions.

    Parameters
    ----------
    filename : str
        path to a file
    read : bool, optional
        check if the file can be read
    write : bool, optional
        check if a file has write permission
    execute : bool, optional
        check if a file has execute permission
    verbose : bool, optional
        If True output log messages

    Returns
    -------
    bool
        True if the file exists and fulfills all requirements, False
        otherwise.
    """
    badthings = []

    def exitstatus():
        if verbose:
            for msg in badthings:
                log.warning(msg)
        return len(badthings) == 0

    if not isinstance(filename, str):
        badthings.append("not a string: %s" % repr(filename))
        return exitstatus()
    elif not os.path.isfile(filename):
        badthings.append("not a file: %s" % repr(filename))
        return exitstatus()
    if read:
        if not os.access(filename, os.R_OK):
            badthings.append("not readable: %s" % repr(filename))
    if write:
        if not os.access(filename, os.W_OK):
            badthings.append("not writeable: %s" % repr(filename))
    if execute:
        if not os.access(filename, os.X_OK):
            badthings.append("not executable: %s" % repr(filename))
    return exitstatus()


def date2seconds(datestring, dformat='%Y-%m-%dT%H:%M:%S.%f'):
    """
    Convert a header datestring to seconds

    Parameters
    ----------
    datestring : str
    dformat : str
        expected format of the date-time string.  Default format
        is YYYY-MM-DDThh:mm:ss.ff

    Returns
    -------
    float
    """
    if dformat.endswith('.%f'):
        if '.' not in datestring:
            datestring += '.0'
    try:
        d = datetime.strptime(datestring, dformat)
        return time.mktime(d.timetuple())
    except ValueError:
        return None


def str_to_value(text):
    """
    Convert a string to an int or float.  If the format is not recognized,
    the original input will be returned.

    Parameters
    ----------
    text : str
        May be an integer (e.g. "123"), a decimal ("1.23"), or
        scientific notation ("1.2e-3").

    Returns
    -------
    int or float

    Examples
    --------
    >>> from sofia_redux.toolkit.utilities.func import str_to_value
    >>> print(str_to_value("4.32e-1"))
    0.432
    """
    regex = r'^(-)?([0-9])+(\.([0-9])+)?([eE](\+|-)?([0-9])+)?$'
    v = text[:]
    if re.match(regex, v):
        if re.search(r'[Ee.]', v):
            return float(v)
        else:
            return int(v)
    else:
        return v


def slicer(array, axis, index, ind=False):
    """
    Returns a slice of an array in arbitrary dimension.

    Parameters
    ----------
    array : numpy.ndarray
        array to slice
    axis : int or array_like
        axis to slice on
    index : int or array_like of int
        index retrieved
    ind : bool, optional
        If True, return the slices rather than sliced array

    Returns
    -------
    numpy.ndarray or tuple of slice
    """
    if isinstance(index, int):
        idx = [slice(None)] * axis
        idx += [index]
        idx += [slice(None)] * (array.ndim - axis - 1)
        idx = tuple(idx)
    else:
        idx = list(index)
        idx.insert(axis, slice(None))
        idx = tuple(idx)

    if ind:
        return idx
    else:
        return array[idx]


def setnumber(value, minval=None, maxval=None, default=1, dtype=int):
    """Sets a value to a valid number type"""
    if not isinstance(value, (int, float)):
        value = default
    if minval is not None and value < minval:
        value = minval
    if maxval is not None and value > maxval:
        value = maxval
    return dtype(value)


def gaussian_model(x, x0, amplitude, fwhm, y0):
    """
    Gaussian model for curve_fit

    Parameters
    ----------
    x : array_like of float
    x0 : float
    amplitude : float
    fwhm : float
    y0 : float

    Returns
    -------
    array_like of float
    """
    sigma = gaussian_fwhm_to_sigma * fwhm
    return amplitude * np.exp((-(x - x0) ** 2) / (2 * (sigma ** 2))) + y0


def to_array_shape(value, shape, dtype=None):
    """
    Broadcast an array to the desired shape.

    Converts an array or value to a numpy.ndarray broadcasting in
    the reverse order of the desired shape.  For example, if `shape`
    is equal to (1, 2, 3, 4), then you may broadcast any of the
    following shapes to the desired shape::

        (4,)
        (3, 4)
        (2, 3, 4)
        (1, 2, 3, 4)

    Parameters
    ----------
    value
        The value to broadcast
    shape : array_like of int
        The desired shape
    dtype : type, optional
        Convert the final array to a specific data type.

    Returns
    -------
    numpy.ndarray
        Output array of shape `shape`.
    """
    if value is None:
        return
    if not hasattr(shape, '__len__'):
        shape = [shape]
    if not hasattr(value, '__len__') or isinstance(value, (str, int, float)):
        value = np.full(shape, value)
    value = np.array(value)
    if dtype is not None:
        try:
            value = value.astype(dtype)
        except ValueError:
            log.error("unable to convert to %s" % dtype)
            return
    s = tuple(shape)
    if value.shape == s:
        return value
    ndim = len(s)
    if value.ndim > ndim:
        log.error('value must have less dimensions than desired shape')
        return
    diff = ndim - value.ndim
    if s[diff:] != value.shape:
        log.error("incompatible broadcasting shapes")
        return
    for adddim in s[diff - 1::-1]:
        value = np.repeat([value], adddim, axis=0)
    return value


def recursive_dict_update(original, new):
    """
    Recursively update a dictionary

    Will update a nested dictionary with new values.

    Parameters
    ----------
    original : dict
        The dictionary to update.
    new : dict
        Dictionary containing new values.

    Returns
    -------
    dict
        The original dictionary updated with new values.

    Examples
    --------
    >>> from sofia_redux.toolkit.utilities.func import recursive_dict_update
    >>> d1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>> d2 = {'b': {'d': 4}}
    >>> dnew = recursive_dict_update(d1, d2)
    >>> print(dnew)
    {'a': 1, 'b': {'c': 2, 'd': 4}}
    """
    for k, v in new.items():
        if isinstance(v, Mapping):
            original[k] = recursive_dict_update(original.get(k, {}), v)
        else:
            original[k] = v
    return original


def stack(*samples, copy=True):
    values = np.asarray(samples[0], dtype=float)
    shape = values.shape
    n_stack = len(samples)
    v = np.empty((n_stack, values.size), dtype=float)
    v[0] = values.ravel()
    for dim in range(1, n_stack):
        iv = np.asarray(samples[dim], dtype=float)
        if iv.shape != shape:
            raise ValueError(
                "samples[0] and samples[%i] shape mismatch %s != %s" %
                (dim, repr(shape), repr(iv.shape)))
        v[dim] = iv.ravel()
    return v.copy() if copy else v


def faststack(*samples):
    test = np.asarray(samples[0])
    result = np.empty((len(samples), test.size), dtype=test.dtype)
    result[0] = test.ravel()
    for i, s in enumerate(samples[1:]):
        result[i + 1] = np.asarray(s).ravel()
    return result


def taylor(order, n):
    """
    Taylor expansion generator for Polynomial exponents

    Parameters
    ----------
    order : int
        Order of Polynomial
    n : int
        Number of variables to solve for

    Yields
    ------
    n-tuple of int
        The next polynomial exponent
    """
    if n == 0:
        yield()
        return
    for i in range(order + 1):
        for result in taylor(order - i, n - 1):
            yield (i,) + result


def bytes_string(size_bytes):
    """
    Convert a number of bytes to a string with correct suffix

    Parameters
    ----------
    size_bytes : float or int

    Returns
    -------
    str
       Formated as x.xxNB where x represents an integer and N is
       one of {B, K, M, G, T, P, E, Z, Y}

    Examples
    --------
    >>> from sofia_redux.toolkit.utilities.func import bytes_string
    >>> print(bytes_string(2e10))
    18.63GB
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(math.log(size_bytes, 1024)))
    power = 1024 ** i
    size = round(size_bytes / power, 2)
    return "%s%s" % (size, size_name[i])


def remove_sample_nans(samples, error, mask=False):
    """
    Remove any samples containing NaNs from sample points

    Parameters
    ----------
    samples : numpy.ndarray (ndim + 1, npoints)
    error : numpy.ndarray (npoints,), optional
    mask : bool, optional
        If True, return a mask of valid (True) data points

    Returns
    -------
    (samples, error) or mask
        inputs without any NaNs if mask is False, otherwise a
        mask (npoints,) marking valid (True) data points.
    """
    select = np.isnan(samples)
    ndim = samples.ndim

    if ndim == 2:
        select = np.any(select, axis=0)

    doerr = isinstance(error, np.ndarray)
    if doerr:
        select |= (error == 0) | (np.isnan(error))
    np.logical_not(select, out=select)
    if mask:
        return select

    if ndim == 2:
        samples = samples[:, select]
    else:
        samples = samples[select]
    if doerr:
        error = error[select]
    return samples, error


def bitset(arr, bits, skip_checks=False):
    """
    Return a byte array the same size as the input array.

    A pixel is set if any of the bits requested are set in arr.
    Uses the Gumley ishft technique.

    Parameters
    ----------
    arr : array_like of int
        (shape,) the array to search
    bits : int or array_like of int
        The bits to search.  Note that the "first" bit is denoted as
        zero, while the "second" bit is denoted as 1.
    skip_checks : bool, optional
        Do not perform any error checking

    Returns
    -------
    numpy.ndarray of bool:
        (shape,) The pixel is set if any of the bits requested are set
        in array.
    """
    if not skip_checks:
        bits = np.asarray(bits, dtype=int)
        arr = np.asarray(arr, dtype=int)
        if bits.shape == ():
            bits = bits[None]
        if arr.shape == ():
            arr = arr[None]

    result = np.full(arr.shape, 0)
    for bit in bits:
        tmp = (arr >> bit) & 1
        idx = result < tmp
        result[idx] = tmp[idx]

    return result


def nantrim(xin, flag=0, trim=False, bounds=False):
    """
    Derive a mask to trim NaNs from an array

    Parameters
    ----------
    xin : array_like of float (shape)
        The n-D array to be trimmed
    flag : int, optional
        0 -> trailing NaNs are removed
        1 -> leading NaNs are removed
        2 -> leading and trailing NaNs are removed
        3 -> all NaNs are removed
    trim : bool, optional
        If True, apply the mask and return the trimmed array.  Note that
        if flag == 3, the returned array will be flattened.  Overidden by
        `bounds`.
    bounds : bool, optional
        If True, return the start and end index points for each dimension
        of `xin`.  The output array shape will be (2, ndim) where
        result[0, 1] would give the start index of the second dimension
        and result[1, 1] would give the end index of the second dimension.
        Note that end points give the indices one would supply to slices,
        i.e. an endpoint of 10 would mean indices < 10 are valid,
        but one should trim indices >= 10.  `Bounds` overrides `trim`

    Returns
    -------
    numpy.ndarray
        A bool array (shape) where True indicates an element to keep, and
        False indicates a value to remove.  If `trim` is set to True then
        the output will be a copy of the trimmed array.
    """
    if flag not in [0, 1, 2, 3]:
        raise ValueError("unknown flag")
    valid = ~np.isnan(xin)
    ndim, shape = valid.ndim, valid.shape

    ends = np.zeros((2, ndim), dtype=int)
    ends[1] = shape

    if bounds and ndim and flag == 3:
        raise ValueError(
            "Cannot determine bounds for an n-D array with flag=3")

    elif flag == 3 or not hasattr(valid, '__len__') or \
            valid.all() or not valid.any():
        if bounds:
            if not valid.all():
                ends[1] = 0
            return ends if ndim > 1 else ends[:, 0]
        elif trim:
            return np.asarray(xin).copy()[valid]
        else:
            return valid

    for dimi, n in enumerate(shape):
        if flag in [1, 2]:  # leading NaNs
            naninds = np.asarray(np.argmax(valid, axis=dimi))
            naninds = naninds[naninds != 0]
            if naninds.size != 0:
                ends[0, dimi] = naninds.min()
        if flag in [0, 2]:  # trailing NaNs
            naninds = np.asarray(
                np.argmax(np.flip(valid, axis=dimi), axis=dimi))
            naninds = naninds[naninds != 0]
            if naninds.size != 0:
                ends[1, dimi] = n - naninds.min()

    if bounds:
        return ends if ndim > 1 else ends[:, 0]

    size = np.ptp(ends, axis=0)
    inds = np.meshgrid(
        *(np.arange(size[i], dtype=int) + ends[0, i]
          for i in range(ndim)), sparse=True, indexing='ij')

    if trim:
        return np.asarray(xin).copy()[tuple(inds)]
    else:
        mask = np.full(shape, False)
        mask[tuple(inds)] = True
        return mask


def julia_fractal(sy, sx, c0=-0.4, c1=0.6, iterations=256,
                  xrange=(-1, 1), yrange=(-1, 1), normalize=True):
    """
    Generate a 2-D Julia fractal image

    Parameters
    ----------
    sy : int
        y dimension size.
    sx : int
        x dimension size.
    c0 : float, optional
        The c0 coefficient.
    c1 : float, optional
        The c1 coefficient.
    iterations : int, optional
        The number of steps.
    xrange : array_like of int or float, optional
        The range of x values.
    yrange : array_like of int or float, optional
        The range of y values.
    normalize : bool, optional

    Returns
    -------

    """
    x = np.linspace(xrange[0], xrange[1], sx)[None]
    y = np.linspace(yrange[0], yrange[1], sy)[..., None]
    z = np.tile(x, (sy, 1)) + 1j * np.tile(y, (1, sx))
    c = np.full((sy, sx), c0 + 1j * c1)
    mask = np.full((sy, sx), True)
    result = np.zeros((sy, sx))

    for i in range(iterations):
        z[mask] *= z[mask]
        z[mask] += c[mask]
        mask[np.abs(z) > 2] = False
        result[mask] = i

    if normalize:
        result /= result.max()

    return result


def nansum(a, axis=None, dtype=None, out=None, keepdims=0, missing=np.nan):
    """
    Emulates the behaviour of np.nansum for NumPy versions <= 1.9.0.

    Returns NaN if all elements of `a` are NaN rather than zero.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the sum is computed. The default is to compute
        the sum of the flattened array.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  By default, the dtype of `a` is used.  An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either
        (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        bits. For inexact inputs, dtype must be inexact.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``. If provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.  See
        :ref:`ufuncs-output-type` for more details. The casting of NaN to
        integer can yield unexpected results.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        If the value is anything but the default, then
        `keepdims` will be passed through to the `mean` or `sum` methods
        of sub-classes of `ndarray`.  If the sub-classes methods
        does not implement `keepdims` any exceptions will be raised.
    missing : int or float, optional
        The value to replace all NaN slices with.  The default is NaN.

    Returns
    -------
    nansum : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which it is returned. The result has the same
        size as `a`, and the same shape as `a` if `axis` is not None
        or `a` is a 1-d array.
    """
    a, mask = _replace_nan(a, 0)
    if mask is None:
        return np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    mask = np.all(mask, axis=axis, keepdims=keepdims)
    tot = np.sum(a, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    if np.any(mask):
        tot = _copyto(tot, missing, mask)
    return tot


def remove_files(folder):  # pragma: no cover
    """
    Delete all files in a given folder.

    Parameters
    ----------
    folder : str
        The file path to the folder.

    Returns
    -------
    None
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as err:
            print(f"remove_files: Failed on filepath {file_path}: {err}")


def clear_numba_cache(module=None):  # pragma: no cover
    """
    Delete the Numba cache for the sofia_redux toolkit.

    Sometimes Numba refuses to acknowledge that an update has been made to a
    function and will continue to reference a previously cached version.  This
    function clears the cache so that the next time any Numba function is
    called, the new version will be used.

    Returns
    -------
    None
    """
    if module is None:
        module = toolkit_module
    root_folder = os.path.realpath(os.path.dirname(module.__file__))

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    remove_files(os.path.join(root, dirname))
                except Exception as err:
                    print(f"clear_numba_cache: Failed on {root}: {err}")


def byte_size_of_object(obj):
    """
    Return the size of a Python object in bytes.

    Parameters
    ----------
    obj : object

    Returns
    -------
    byte_size : int
    """
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr
                    if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz
