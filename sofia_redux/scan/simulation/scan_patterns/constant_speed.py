# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D

__all__ = ['to_constant_speed']


def to_constant_speed(pattern):
    """
    Convert a pattern such that the samples are equally spaced wrt speed.

    Since this uses linear interpolation, please ensure there are sufficient
    samples to adequately represent the scanning pattern.

    Parameters
    ----------
    pattern : Coordinate2D
        The scanning pattern offsets that should be resampled to a constant
        speed.

    Returns
    -------
    pattern : Coordinate2D
        The elevation/cross-elevation scan pattern sampled at `t_interval`
        in the units of `radius`.
    """
    dx = pattern.x * 0
    dy = pattern.y * 0
    dx[1:] = pattern.x[1:] - pattern.x[:-1]
    dy[1:] = pattern.y[1:] - pattern.y[:-1]
    dr = np.hypot(dx, dy)
    mean_dr = np.mean(dr[1:])

    relative_spacing = dr / mean_dr
    if isinstance(relative_spacing, units.Quantity):
        relative_spacing = relative_spacing.decompose().value

    cumulative_spacing = np.cumsum(relative_spacing)
    apparent_spacing = np.arange(cumulative_spacing.size)
    x = np.interp(apparent_spacing, cumulative_spacing, pattern.x)
    y = np.interp(apparent_spacing, cumulative_spacing, pattern.y)
    return Coordinate2D([x, y])
