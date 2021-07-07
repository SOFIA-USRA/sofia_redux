# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.toolkit.utilities.func import nantrim
from sofia_redux.toolkit.stats import meancomb
from sofia_redux.spectroscopy.combflagstack import combflagstack
from sofia_redux.spectroscopy.interpflagspec import interpflagspec
from sofia_redux.spectroscopy.interpspec import interpspec
import warnings

__all__ = ['mergespec']


def mergespec(spec1, spec2, info=None, sum_flux=False):
    """
    Combine two spectra into a single spectrum

    The two arrays are combined along the independent (x) values (row 0
    of both arrays).  If the x-range of `spec2` overlaps with that of
    `spec1`, then the dependent (row 1), error (row 2), and bitmask (row 3)
    of `spec2` is interpolated onto `spec1` and combined appropriately::

        |------------| array 1
                |--------------| array 2
        |-------+++++----------| array 2 merged with array 1

    In the combined example above, + indicates a combined point and -
    indicates a original point from array 1 or 2.

    If the x-ranges of `spec1` and `spec2` do not overlap, then arrays will
    simply be joined together.  Note that the final output array is
    sorted according to x (lowest -> highest).  If the arrays do not
    overlap then the separation between the arrays are marked by two NaNs
    in all rows > 0: one at the last element of the lowest x-ranged array
    and one as the first element of the highest x-ranged array::

        |----------o-| array 1
                       |-o----------| array 2
        |----------oxxo----------| array 2 merged with array 1

    In the separated example above, x indicates a NaN and - indicates an
    original point from array 1 or 2, and 'o' marks the penultimate or
    second element.

    Note that all NaNs will be trimmed from the beginning and end of the
    input arrays

    Parameters
    ----------
    spec1 : array_like of float
        (2-4, N) array matching the shape of `spec2` where:
            spec1[0] = independent variable
            spec1[1] = dependent variable
            spec1[2] = error in dependent variable, optional
            spec1[3] = bitmask
    spec2 : array_like of float
        (2-4, N) array matching the shape of `spec1` where:
            spec2[0] = independent variable
            spec2[1] = dependent variable
            spec2[2] = error in dependent variable, optional
            spec2[3] = bitmask
    info : dict, optional
        If supplied will be updated with:
            overlap_range -> numpy.ndarray (2,)
                The (min, max) wavelength range over which arrays overlap.
    sum_flux : bool, optional
        If True, sum the flux instead of averaging.

    Returns
    -------
    numpy.ndarray of numpy.float64
        (2-4, M) array where row index gives the following values:
            0: combined independent variable
            1: combined dependent variable
            2: combined error
            3: combined bit-mask
    """
    spec1, spec2 = np.array(spec1), np.array(spec2)
    shape = spec1.shape

    if spec1.ndim != 2 or spec2.ndim != 2:
        raise ValueError("spec1 and spec2 must have 2 dimensions")
    elif shape[0] < 2:
        raise ValueError("spec1 and spec2 must have 2 or more rows")
    elif shape[0] != spec2.shape[0]:
        raise ValueError("spec1 and spec2 must have equal rows")

    doerr = shape[0] >= 3
    dobit = shape[0] >= 4

    # Trim NaNs from the beginning and ends of the spectrum
    spec1 = spec1[:, nantrim(spec1[1], 2)]
    spec2 = spec2[:, nantrim(spec2[1], 2)]
    range1 = np.nanmin(spec1[0]), np.nanmax(spec1[0])
    range2 = np.nanmin(spec2[0]), np.nanmax(spec2[0])

    overlapped, overlap_range = False, [np.nan, np.nan]
    overlap = np.empty((shape[0], 0))
    overlap2 = (spec2[0] >= range1[0]) & (spec2[0] <= range1[1])
    if overlap2.any():
        # Interpolate dependent values (and error) from spec2 onto spec1
        ie2 = spec2[2, overlap2] if doerr else None
        if2 = interpspec(spec2[0, overlap2], spec2[1, overlap2],
                         spec1[0], error=ie2, leavenans=True, cval=np.nan)
        if2, ie2 = if2 if doerr else (if2, None)

        overlap1 = nantrim(if2, 2)
        overlapped = overlap1.any()
        if overlapped:
            overlap = np.full((shape[0], overlap1.sum()), np.nan)
            overlap[0] = spec1[0][overlap1]
            if2 = [spec1[1][overlap1], if2[overlap1]]
            if doerr:
                ie2 = np.array([spec1[2][overlap1], ie2[overlap1]]) ** 2

            # combine dependent values (and error)
            if sum_flux:
                overlap[1] = np.sum(if2, axis=0)
                if doerr:
                    overlap[2] = np.sqrt(np.sum(ie2, axis=0))
            else:
                if2 = meancomb(if2, variance=ie2, ignorenans=False,
                               axis=0, returned=doerr)
                if doerr:
                    overlap[1] = if2[0]
                    overlap[2] = np.sqrt(if2[1])
                else:
                    overlap[1] = if2

            # interpolate bit-flags from spec2 to spec1 and then combine
            if dobit:
                ib2 = interpflagspec(
                    spec2[0, overlap2], spec2[3, overlap2], spec1[0])
                overlap[3] = combflagstack(
                    [spec1[3, overlap1], ib2[overlap1]])

            overlap_range = [np.nanmin(overlap[0]), np.nanmax(overlap[0])]
            sortw = np.argsort(overlap[0])
            overlap = overlap[np.arange(shape[0])[:, None], sortw[None]]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        highcut = np.min([range1[1], range2[1]])
        lowcut = np.max([range1[0], range2[0]])
        if range1[0] < range2[0]:
            left = spec1[:, spec1[0] < lowcut]
        else:
            left = spec2[:, spec2[0] < lowcut]
        if range1[1] > range2[1]:
            right = spec1[:, spec1[0] > highcut]
        else:
            right = spec2[:, spec2[0] > highcut]

    if left.shape[1] > 1:
        sortw = np.argsort(left[0])
        left = left[np.arange(shape[0])[:, None], sortw[None]]
        if not overlapped:
            left[1:, -1] = np.nan
        overlap = np.concatenate((left, overlap), axis=1)

    if right.shape[1] > 1:
        sortw = np.argsort(right[0])
        right = right[np.arange(shape[0])[:, None], sortw[None]]
        if not overlapped:
            right[1:, 0] = np.nan
        overlap = np.concatenate((overlap, right), axis=1)

    if isinstance(info, dict):
        info['overlap_range'] = np.array(overlap_range)

    return overlap
