# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)
from sofia_redux.scan.simulation.scan_patterns.constant_speed import (
    to_constant_speed)

__all__ = ['daisy_pattern_offset', 'daisy_pattern_equatorial']


def daisy_pattern_offset(radius, radial_period, t_interval,
                         n_oscillations=22, constant_speed=False):
    """
    Create a daisy scan pattern in offset coordinates.

    The daisy pattern has the following form in two dimensions::

      x = r.sin(a).cos(b)
      y = r.sin(a).sin(b)
      a = 2.pi.t/radial_period + radial_phase
      b = 2.t/radial_period + rotation_phase
      rotation_phase = 2.n_oscillations
      radial_phase = pi.rotation_phase

    The time (t) is generated in the range 0 -> n_oscillations * radial_period
    in increments ot t_interval.  If `constant_speed` is `False`, the speed
    of the pattern will be minimal near the edges of the petals and maximal
    near the center.  Due to the fact that the radial and rotational phases
    are auto-generated such that the pattern forms a closed shape, the samples
    will also occur along increasingly separated concentric rings from the
    origin.  Therefore, for better scanning coverage, `constant_speed` may
    be set to `True` so that the physical distance between each sample are
    approximately equal.

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
    constant_speed : bool, optional
        If `True`, return a pattern where the speed between each sample point
        is equal.

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

    pattern = Coordinate2D(np.stack([dx, dy]))
    if constant_speed:
        pattern = to_constant_speed(pattern)

    return pattern


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
    if 'constant_speed' in kwargs:
        constant_speed = kwargs['constant_speed']
    else:
        constant_speed = False

    equatorial_offset = daisy_pattern_offset(
        radius, radial_period, t_interval, n_oscillations=n_oscillations,
        constant_speed=constant_speed)
    equatorial = EquatorialCoordinates(center)
    equatorial.add(equatorial_offset)
    return equatorial
