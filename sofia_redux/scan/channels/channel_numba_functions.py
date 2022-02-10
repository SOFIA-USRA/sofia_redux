# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np
from sofia_redux.scan.utilities import numba_functions

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['flag_weights', 'get_typical_gain_magnitude',
           'get_one_over_f_stat', 'get_source_nefd']


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def flag_weights(channel_gain, channel_weight, channel_dof, channel_flags,
                 min_weight, max_weight, exclude_flag, dof_flag,
                 sensitivity_flag, default_weight):  # pragma: no cover
    """
    Flag channels according to degrees-of-freedom and weight.

    Channels will be flagged with the DOF (degrees-of-freedom) flag if its
    degrees-of-freedom is <= 0, or unflagged otherwise.  Channels will also
    be flagged for sensitivity if::

       wg2 < m * min_weight
       wg2 > m * max_weight

    where wg2 = channel_weight * channel_gain^2, and::

       m = exp(inner_80_percent_mean(ln(1 + wg2))) - 1

    Channels that fall within the above range are likewise unflagged for
    sensitivity.  Any channels that are fully or partially marked with the
    exclude flag will not be included in the mean calculation, but may have
    flags set.  The sum of wg2 for all zero flagged channels will be returned.

    Parameters
    ----------
    channel_gain : numpy.ndarray (float)
        The gains for each channel of shape (n_channels,).
    channel_weight : numpy.ndarray (float)
        The weights for each channel of shape (n_channels,).  Weights are
        equivalent to 1/variance.
    channel_dof : numpy.ndarray (float)
        The degrees-of-freedom for each channel of shape (n_channels,).
    channel_flags : numpy.ndarray (int)
        The flags for each channel of shape (n_channels,).  Flags will be
        updated in place.
    min_weight : float
        The minimum acceptable (1/variance) weight value.
    max_weight : float
        The maximum acceptable (1/variance) weight value.
    exclude_flag : int
        An integer flag marking channel flag types that should be excluded from
        the flagging, or overall weight/gain calculation.
    dof_flag : int
        The integer marking channels flagged as having insufficient degrees-of-
        freedom.
    sensitivity_flag : int
        The integer marking channels flagged as having unacceptable weight
        (variance) values based on whether it falls outside the `min_weight` to
        `max_weight` range.
    default_weight : float
        The default weight value for channels.  Any channel with a weight value
        equivalent to the default value will not be flagged or included in
        any calculations, as channels with this weight value will not have been
        processed due to other factors.

    Returns
    -------
    n_points, weight_sum, channel_flags : int, float, numpy.ndarray (int)
       `n_points` are the number of channels that have valid `dof` values,
       are not excluded by `exclude_flag`, and have positive non-default weight
       values.  `weight_sum` is sum(weight * gain^2) for all zero-flagged
       channels (following unflagging), and `channel_flags` are the updated
       channel flags.
    """
    n_channels = channel_gain.size
    weights = np.empty(n_channels, dtype=nb.float64)
    valid_points = 0
    for i in range(n_channels):
        # Unflag/flag Degrees-of-freedom flag
        if channel_dof[i] > 0:
            if (channel_flags[i] & dof_flag) != 0:
                channel_flags[i] ^= dof_flag
        else:
            channel_flags[i] |= dof_flag
            continue

        if (channel_flags[i] & exclude_flag) != 0:
            continue

        gain = channel_gain[i]
        if gain == 0:
            continue
        weight = channel_weight[i]
        if (weight <= 0) or (weight == default_weight):
            continue
        weights[valid_points] = np.log(1.0 + (weight * (gain ** 2)))
        valid_points += 1

    if valid_points == 0:
        return valid_points, 0.0, channel_flags

    if valid_points > 10:
        mean_log_weight = numba_functions.robust_mean(
            weights[:valid_points], tails=0.1)
    else:
        mean_log_weight = np.nanmedian(weights[:valid_points])

    mean_weight = np.exp(mean_log_weight) - 1
    lower_weight = min_weight * mean_weight
    upper_weight = max_weight * mean_weight

    sum_weight = 0.0
    for i in range(n_channels):
        weight = channel_weight[i]
        wg2 = weight * (channel_gain[i] ** 2)
        if (wg2 < lower_weight) or (wg2 > upper_weight):
            channel_flags[i] |= sensitivity_flag
        elif weight == default_weight:
            channel_flags[i] |= sensitivity_flag
        else:
            if (channel_flags[i] & sensitivity_flag) != 0:
                channel_flags[i] ^= sensitivity_flag
            if channel_flags[i] == 0:
                sum_weight += wg2

    return valid_points, sum_weight, channel_flags


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_typical_gain_magnitude(gains):  # pragma: no cover
    """
    Return the typical gain magnitude for a gain array.

    The typical gain magnitude (g_mag) for an array of gains (g) is given by::

        g_mag = exp(inner_80_percent_mean(ln(1 + |g|))) - 1

    Only finite values of g will be included in the mean calculation.  If g_mag
    evaluates to < 0, then a result of 1 will be returned (which should not
    happen).  Note that this mean will rarely ever evaluate to the true mean.
    Thus, removing this mean and re-evaluating (as is done with many functions
    for sofia_scan) will probably get you close to zero, but not actually hit
    the mark.  Just be aware of this as it can cause confusion.

    Parameters
    ----------
    gains : numpy.ndarray (float)

    Returns
    -------
    gain_magnitude : float
    """
    n = gains.size
    log_p1_gains = np.empty(n, dtype=nb.float64)
    valid = 0
    for i in range(n):
        gain = gains[i]
        if not np.isfinite(gain):
            continue
        log_p1_gains[valid] = np.log(1.0 + abs(gain))
        valid += 1

    log_p1_mean = numba_functions.robust_mean(log_p1_gains[:valid], tails=0.1)
    if not np.isfinite(log_p1_mean):
        return 1.0
    average_gain = np.exp(log_p1_mean) - 1.0
    if average_gain > 0:
        return average_gain
    else:  # pragma: no cover
        return 1.0


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_one_over_f_stat(weights, one_over_f_stats, flags):  # pragma: no cover
    """
    Return the total channel 1/f statistic.

    The 1/f statistic is given as:

        f1_stat = sqrt(sum(w * f^2) / sum(w))

    where w are the channel weights and f are the channel 1/f statistics.

    Parameters
    ----------
    weights : numpy.ndarray (float)
        The channel weights of shape (n_channels,).
    one_over_f_stats : numpy.ndarray (float)
        The channel 1/f statistics of shape (n_channels,).
    flags : numpy.ndarray (int)
        The channel flags of shape (n_channels,).  Only zero-valued flags will
        be included in the calculations.

    Returns
    -------
    f1_stat : float
    """
    wd_sum = 0.0
    w_sum = 0.0
    for i in range(weights.size):
        if flags[i] != 0:
            continue
        w = weights[i]
        if w == 0:
            continue
        f = one_over_f_stats[i]
        if np.isnan(f):
            continue
        wd_sum += w * f * f
        w_sum += w

    if w_sum <= 0:
        return np.nan
    return np.sqrt(wd_sum / w_sum)


@nb.njit(cache=True, nogil=False, parallel=False, fastmath=False)
def get_source_nefd(filtered_source_gains, weight, variance, flags,
                    integration_time, integration_gain):  # pragma: no cover
    """
    Return the source Noise-Equivalent-Flux-Density (NEFD).

    The source NEFD is given as:

    nefd = sqrt(n * t / sum(g^2 / v)) / abs(integration_gain)

    where n are the number of mapping channels (unflagged and positive
    weights), t is the integration time, g is the source gain, and v is
    the channel variance.

    Parameters
    ----------
    filtered_source_gains : numpy.ndarray (float)
        The filtered source gains of shape (n_channels,) which is the product
        of channel coupling, gain, and source filtering.
    weight : numpy.ndarray (float)
        The channel weights of shape (n_channels,).
    variance : numpy.ndarray (float)
        The channel variances of shape (n_channels,).
    flags : numpy.ndarray (int)
        The channel flags of shape (n_channels,) where non-zero values are
        not included in the calculation.
    integration_time : float
        The integration time in arbitrary units (numba).  Remember to reconvert
        after.
    integration_gain : float
        The integration gain factor.

    Returns
    -------
    nefd : float
        The source NEFD
    """
    mapping_channels = 0
    sum_pw = 0.0
    n_channels = filtered_source_gains.size
    abs_gain = np.abs(integration_gain)
    for i in range(n_channels):
        if flags[i] != 0:
            continue
        if weight[i] <= 0:
            continue
        mapping_channels += 1
        g = filtered_source_gains[i]
        if g == 0:
            continue
        # Variance should be non-zero since weight already checked
        sum_pw += g * g / variance[i]

    if sum_pw == 0:
        return np.inf
    else:
        return np.sqrt(integration_time * mapping_channels / sum_pw) / abs_gain
