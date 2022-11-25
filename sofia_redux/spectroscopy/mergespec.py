# Licensed under a 3-clause BSD style license - see LICENSE.rst

import warnings

from astropy import log
import numba as nb
import numpy as np

from sofia_redux.toolkit.utilities.func import nantrim
from sofia_redux.toolkit.stats import meancomb
from sofia_redux.spectroscopy.combflagstack import combflagstack
from sofia_redux.spectroscopy.interpflagspec import interpflagspec
from sofia_redux.spectroscopy.interpspec import interpspec

__all__ = ['mergespec']


def mergespec(spec1, spec2, info=None, sum_flux=False, s2n_threshold=None,
              s2n_statistic='median', noise_test=False, local_noise=False,
              local_radius=3):
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
    s2n_threshold : float, optional
        If set and the value is greater than 0, and errors are provided,
        data below this value times the reference signal-to-noise (S/N)
        value in the spectrum will be clipped before combination.
    s2n_statistic : {'median', 'mean', 'max'}, optional
        Statistic to use for computing the reference S/N value.
        Default is median.
    noise_test : bool, optional
        If set, only the noise is considered for thresholding spectra.
        The s2n_threshold is interpreted as a fraction of 1/noise.
    local_noise : bool, optional
        If set, noise for the spectrum is computed from the standard deviation
        in a sliding window with radius local_radius.
    local_radius : int, optional
        Sets the local window in pixels for computing noise if local_noise
        is set.

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

    if s2n_statistic == 'mean':
        func = np.nanmean
    elif s2n_statistic == 'max':
        func = np.nanmax
    else:
        func = np.nanmedian

    doerr = shape[0] >= 3
    dobit = shape[0] >= 4

    if doerr:
        # Check for zero-error data
        bad_error = ~np.isfinite(spec1[2]) | (spec1[2] == 0)
        spec1[1][bad_error] = np.nan
        spec1[2][bad_error] = np.nan
        bad_error = ~np.isfinite(spec2[2]) | (spec2[2] == 0)
        spec2[1][bad_error] = np.nan
        spec2[2][bad_error] = np.nan

    # Trim NaNs from the beginning and ends of the spectrum
    spec1 = spec1[:, nantrim(spec1[1], 2)]
    spec2 = spec2[:, nantrim(spec2[1], 2)]
    range1 = np.nanmin(spec1[0]), np.nanmax(spec1[0])
    range2 = np.nanmin(spec2[0]), np.nanmax(spec2[0])

    # get good S/N range if desired
    ok_s2n, s2n_1, s2n_2 = None, None, None
    if doerr and s2n_threshold is not None and s2n_threshold > 0:
        if local_noise:
            log.debug('Using sliding standard deviation for noise')
            noise_1 = _sliding_stddev(spec1[1], local_radius)
            noise_2 = _sliding_stddev(spec2[1], local_radius)
        else:
            log.debug('Using input error for noise')
            noise_1 = spec1[2]
            noise_2 = spec2[2]

        if noise_test:
            log.debug(f'Threshold statistic is {func} on 1/N')
            s2n_1 = 1 / noise_1
            s2n_2 = 1 / noise_2
        else:
            log.debug(f'Threshold statistic is {func} on S/N')
            s2n_1 = spec1[1] / noise_1
            s2n_2 = spec2[1] / noise_2

        # compute threshold from equal sized portions at ends
        # of spectra
        ok_s2n = [s2n_threshold * func(s2n_1),
                  s2n_threshold * func(s2n_2)]
        log.debug(f'S/N threshold: {ok_s2n}')
        if not (ok_s2n[0] > 0 and ok_s2n[1] > 0):
            log.warning(f'Bad S/N; ignoring threshold '
                        f'values {ok_s2n}')
            ok_s2n = None

    overlapped, overlap_range = False, [np.nan, np.nan]
    overlap = np.empty((shape[0], 0))
    overlap2 = (spec2[0] >= range1[0]) & (spec2[0] <= range1[1])
    if overlap2.any():
        # Interpolate dependent values (and error) from spec2 onto spec1
        ie2 = spec2[2, overlap2] if doerr else None
        if2 = interpspec(spec2[0, overlap2], spec2[1, overlap2],
                         spec1[0], error=ie2, leavenans=True, cval=np.nan)
        if2, ie2 = if2 if doerr else (if2, None)

        if ok_s2n is not None:
            i_s2n_2 = interpspec(spec2[0, overlap2], s2n_2[overlap2],
                                 spec1[0], leavenans=True, cval=np.nan)
        else:
            i_s2n_2 = None

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

            elif ok_s2n is not None:
                use_s1 = ((s2n_1[overlap1] > ok_s2n[0])
                          & (i_s2n_2[overlap1] <= ok_s2n[1]))
                use_s2 = ((s2n_1[overlap1] <= ok_s2n[0])
                          & (i_s2n_2[overlap1] > ok_s2n[1]))
                combined = meancomb(if2, variance=ie2, ignorenans=False,
                                    axis=0, returned=doerr)

                overlap[1] = combined[0]
                overlap[1][use_s1] = if2[0][use_s1]
                overlap[1][use_s2] = if2[1][use_s2]

                overlap[2] = np.sqrt(combined[1])
                overlap[2][use_s1] = np.sqrt(ie2[0][use_s1])
                overlap[2][use_s2] = np.sqrt(ie2[1][use_s2])

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


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def _sliding_stddev(data, radius):  # pragma: no cover
    n = data.size
    r = int(radius)
    result = np.empty(n, dtype=nb.float64)
    for i in range(n):
        start = i - r
        end = i + r
        if start < 0:
            start = 0
        if end > n:
            end = n
        result[i] = np.nanstd(data[start:end])
    return result
