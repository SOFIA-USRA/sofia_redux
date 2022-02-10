# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.channels.channel_numba_functions import (
    flag_weights, get_typical_gain_magnitude, get_one_over_f_stat,
    get_source_nefd)


def test_flag_weights():
    dof_flag = 1
    sensitivity_flag = 2
    exclude_flag = 4
    default_weight = 1.0

    channel_gain = 1 + np.linspace(1, 100, 100) / 100
    channel_weight = np.full(100, 0.01)
    channel_dof = np.ones(100)
    channel_flags = np.zeros(100, dtype=int)

    min_weight = 0.001
    max_weight = 10.0

    # Standard run - no flagging, check dof is unflagged
    channel_flags.fill(dof_flag)
    n_points, w_sum, flags = flag_weights(
        channel_gain=channel_gain,
        channel_weight=channel_weight,
        channel_dof=channel_dof,
        channel_flags=channel_flags,
        min_weight=min_weight,
        max_weight=max_weight,
        exclude_flag=exclude_flag,
        dof_flag=dof_flag,
        sensitivity_flag=sensitivity_flag,
        default_weight=default_weight
    )
    assert n_points == 100 and np.isclose(w_sum, 2.34835)
    assert flags is channel_flags and np.allclose(flags, 0)

    # Check bad degrees of freedom
    bad_dof = np.zeros(100)
    n_points, w_sum, flags = flag_weights(
        channel_gain=channel_gain,
        channel_weight=channel_weight,
        channel_dof=bad_dof,
        channel_flags=channel_flags,
        min_weight=min_weight,
        max_weight=max_weight,
        exclude_flag=exclude_flag,
        dof_flag=dof_flag,
        sensitivity_flag=sensitivity_flag,
        default_weight=default_weight
    )
    assert n_points == 0 and w_sum == 0 and np.allclose(flags, dof_flag)

    # Check exclude flag
    bad_flags = np.full(100, exclude_flag)
    n_points, w_sum, flags = flag_weights(
        channel_gain=channel_gain,
        channel_weight=channel_weight,
        channel_dof=channel_dof,
        channel_flags=bad_flags,
        min_weight=min_weight,
        max_weight=max_weight,
        exclude_flag=exclude_flag,
        dof_flag=dof_flag,
        sensitivity_flag=sensitivity_flag,
        default_weight=default_weight
    )
    assert n_points == 0 and w_sum == 0 and np.allclose(flags, exclude_flag)

    # Check bad weights and gains
    bad_gains = channel_gain.copy()
    bad_gains[:50] = 0.0
    bad_weights = channel_weight.copy()
    bad_weights[50:75] = 0.0
    bad_weights[76:] = default_weight  # leave one good weight
    channel_flags[75] = 2  # check this is unflagged on completion
    n_points, w_sum, flags = flag_weights(
        channel_gain=bad_gains,
        channel_weight=bad_weights,
        channel_dof=channel_dof,
        channel_flags=channel_flags,
        min_weight=min_weight,
        max_weight=max_weight,
        exclude_flag=exclude_flag,
        dof_flag=dof_flag,
        sensitivity_flag=sensitivity_flag,
        default_weight=default_weight
    )
    assert n_points == 1 and np.isclose(w_sum, 0.030976)
    mask = flags == 2
    assert mask.sum() == 99
    assert np.nonzero(~mask)[0][0] == 75

    # Check default weight flagging
    check_default = default_weight + 1e-4
    check_weights = channel_weight.copy()
    check_weights[50] = check_default
    channel_flags.fill(0)
    n_points, w_sum, flags = flag_weights(
        channel_gain=channel_gain,
        channel_weight=check_weights,
        channel_dof=channel_dof,
        channel_flags=channel_flags,
        min_weight=min_weight,
        max_weight=100,
        exclude_flag=exclude_flag,
        dof_flag=dof_flag,
        sensitivity_flag=sensitivity_flag,
        default_weight=check_default
    )
    assert (flags == 2).sum() == 1 and flags[50] == 2
    assert n_points == 99 and np.isclose(w_sum, 2.325549)


def test_get_typical_gain_magnitude():
    g = np.linspace(1.01, 2, 100)
    g_mag = get_typical_gain_magnitude(g)
    assert np.isclose(g_mag, 1.494296828)

    g_bad = np.full(100, np.nan)
    x = get_typical_gain_magnitude(g_bad)
    assert x == 1

    # Check convergence... (doesn't happen with this)
    g -= g_mag
    g_mag = get_typical_gain_magnitude(g)
    assert np.isclose(g_mag, 0.244639943)

    g -= g_mag
    g_mag = get_typical_gain_magnitude(g)
    assert np.isclose(g_mag, 0.281553177)


def test_one_over_f_stat():
    weights = np.ones(100)
    f1 = np.linspace(1.01, 2, 100)
    flags = np.zeros(100, dtype=int)
    f1_stat = get_one_over_f_stat(weights, f1, flags)
    assert np.isclose(f1_stat, 1.5324327065)

    weights[0] = 0.0
    f1[1] = np.nan
    flags[2] = 1
    f1_stat = get_one_over_f_stat(weights, f1, flags)
    assert np.isclose(f1_stat, 1.5455743269)

    weights.fill(0)
    f1_stat = get_one_over_f_stat(weights, f1, flags)
    assert np.isnan(f1_stat)


def test_get_source_nefd():
    integration_time = 9.0
    flags = np.zeros(100, dtype=int)
    filtered_source_gains = np.full(100, 2.0)
    weights = np.full(100, 0.25)
    variance = 1 / weights
    integration_gain = -10.0

    nefd = get_source_nefd(filtered_source_gains=filtered_source_gains,
                           weight=weights,
                           variance=variance,
                           flags=flags,
                           integration_time=integration_time,
                           integration_gain=integration_gain)
    assert nefd == 0.3

    flags[0] = 1
    weights[1] = 0
    filtered_source_gains[2] = 0.0
    nefd = get_source_nefd(filtered_source_gains=filtered_source_gains,
                           weight=weights,
                           variance=variance,
                           flags=flags,
                           integration_time=integration_time,
                           integration_gain=integration_gain)
    assert np.isclose(nefd, 0.3015424266)

    flags.fill(1)
    nefd = get_source_nefd(filtered_source_gains=filtered_source_gains,
                           weight=weights,
                           variance=variance,
                           flags=flags,
                           integration_time=integration_time,
                           integration_gain=integration_gain)
    assert np.isinf(nefd) and nefd > 0
