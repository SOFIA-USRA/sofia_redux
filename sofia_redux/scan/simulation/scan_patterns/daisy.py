# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)

__all__ = ['daisy_pattern_offset', 'daisy_pattern_equatorial']


def daisy_pattern_offset(radius, radial_period, t_interval,
                         n_oscillations=22):
    """
    Create a daisy scan pattern in offset coordinates.

    Parameters
    ----------
    radius : units.Quantity
        The angular radius of the daisy pattern.
    radial_period : units.Quantity
        The time to complete a single radial oscillation.
    t_interval : units.Quantity
        The time interval between sampling points.
    n_oscillations : int or float, optional
        The number of oscillations in the pattern.  The default of 22 gives
        a fully populated area for observation.

    Returns
    -------
    pattern : Coordinate2D
        The elevation/cross-elevation scan pattern sampled at `t_interval`
        in the units of `radius`.
    """
    t_length = n_oscillations * radial_period
    rotation_phase = 2 * n_oscillations * units.Unit('radian')
    radial_phase = rotation_phase * np.pi
    nt = int(np.ceil((t_length / t_interval).decompose().value))
    t = np.arange(nt) * t_interval
    tr = (t / radial_period).decompose().value * units.Unit('radian')
    a = (2 * np.pi * tr) + radial_phase
    b = (2 * tr) + rotation_phase
    dx = radius * np.sin(a) * np.cos(b)
    dy = radius * np.sin(a) * np.sin(b)
    return Coordinate2D(np.stack([dx, dy]))


def daisy_pattern_equatorial(center, t_interval, **kwargs):
    """
    Create a daisy scan pattern in equatorial coordinates.

    The daisy pattern will be a perfect daisy about the equatorial
    coordinates.

    Parameters
    ----------
    center : units.Quantity or EquatorialCoordinates
        The center of the pattern in equatorial coordinates.
    t_interval : units.Quantity
        The sampling interval between output points.

    Returns
    -------
    pattern : EquatorialCoordinates
        The equatorial scan pattern sampled at `t_interval`.
    """
    if 'radius' in kwargs:
        radius = kwargs['radius']
    else:
        radius = 2 * units.Unit('arcmin')
    if 'radial_period' in kwargs:
        radial_period = kwargs['radial_period']
    else:
        radial_period = 5 * units.Unit('second')
    if 'n_oscillations' in kwargs:
        n_oscillations = kwargs['n_oscillations']
    else:
        n_oscillations = 22

    equatorial_offset = daisy_pattern_offset(
        radius, radial_period, t_interval, n_oscillations=n_oscillations)
    equatorial = EquatorialCoordinates(center)
    equatorial.add(equatorial_offset)
    return equatorial
