# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
import numpy as np
from sofia_redux.toolkit.utilities.func import bitset

__all__ = ['combflagstack']


def combflagstack(stack, nbits=8, axis=0):
    """
    Combine bit-set flag arrays.

    Parameters
    ----------
    stack : array_like of int
        (n_stack, mask_shape)  The _stack of bit-set flag arrays to combine.
        The _stack can either be a _stack of spectra (n_spectrum, n_data) or
        a _stack of images (n_images, nrow, ncol).
    nbits : int, optional
        The number of bits that can potentially be set.  This function
        assumes the bits are set sequentially, starting with the zeroth
        bit.  So if nbits is 2, then it will check the 0th and 1st bit.
        The default is to check all eight bits.
    axis : int, optional
        Axis along which to perform the combination.

    Returns
    -------
    numpy.ndarray of int
        The bit-set flag array that reflects the bit-set flags from all
        of the spectra or images.
    """
    stack = np.array(stack)
    if stack.ndim < 2:
        log.error("Invalid _stack dimensions")
        return
    if axis > stack.ndim or axis < 0:
        log.error("Invalid axis")
        return
    axis = int(axis)
    shape = list(stack.shape)
    del shape[axis]

    result = np.zeros(shape, dtype=int)
    for bit in range(nbits):
        bsum = (np.sum(bitset(stack, bit), axis=axis) > 0) * (2 ** bit)
        np.add(result, bsum, out=result)

    np.mod(result, 256, out=result)
    return result
