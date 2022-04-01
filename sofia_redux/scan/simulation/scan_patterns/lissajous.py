# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)
from sofia_redux.scan.simulation.scan_patterns.constant_speed import (
    to_constant_speed)

__all__ = ['lissajous_offset', 'lissajous_pattern_equatorial']


def lissajous_offset(width, height, t_interval, ratio=np.sqrt(2),
                     delta=np.pi * units.Unit('radian') / 2,
                     n_oscillations=20,
                     oscillation_period=10 * units.Unit('second'),
                     constant_speed=False):
    """
    Create a Lissajous scan pattern in offset coordinates.

    The two-dimensions Lissajous pattern is of the following form::

      x = width.sin(a + delta)
      y = height.sin(b + delta)
      a = 2.pi.t/oscillation_period
      b = 2.pi.ratio.t/oscillation_period

    The distance between samples (speed) is maximal at the origin and minimal
    near the edges of the box, which also means the map is under-sampled
    near the center.  An approximation of the pattern using a constant speed
    for all samples can be achieved by setting `constant_speed` to `True`.

    Parameters
    ----------
    width : units.Quantity
        The angular spatial width of the scan.
    height : units.Quantity
        The angular spatial height of the scan.
    t_interval : units.Quantity
        The time interval between sampling points.
    ratio : float, optional
        Determines the number of "lobes" in the lissajous curves relating to
        width/height.  For example 5/4 produces a curve with 5 lobes in the
        x-direction, and 4 in the y-direction.  Irrational numbers result in
        perfect coverage over a long time span.
    delta : units.Quantity, optional
        The angular value giving the apparent rotation as if viewed from a
        third axis.  Any non-zero value results in curve rotated to the
        left-right or up-down depending on `ratio`.
    n_oscillations : int or float, optional
        The number of curve oscillations.
    oscillation_period : units.Quantity, optional
        The time for the curve to complete a single oscillation.  Note that the
        scan length (time) will be `n_oscillations` * `oscillation_period`.
    constant_speed : bool, optional
        If `True`, return a pattern where the speed between each sample point
        is equal.

    Returns
    -------
    pattern : Coordinate2D
        The elevation/cross-elevation scan pattern sampled at `t_interval`.
    """
    fx = 1.0
    fy = ratio * fx

    period_distance = 2 * np.pi * units.Unit('radian')
    scan_time = n_oscillations * oscillation_period
    nt = int(np.ceil((scan_time / t_interval).decompose().value))
    t = np.arange(nt) * t_interval

    rt = period_distance / oscillation_period
    ax = rt * fx * t
    ay = rt * fy * t
    x = width * np.sin(ax + delta) / 2
    y = height * np.sin(ay + delta) / 2
    pattern = Coordinate2D(np.stack([x, y]))
    if constant_speed:
        pattern = to_constant_speed(pattern)

    return pattern


def lissajous_pattern_equatorial(center, t_interval, **kwargs):
    """
    Create a Lissajous scan pattern in equatorial coordinates.

    Please see :func:`lissajous_offset` for a description of the available
    Lissajous parameters.

    Parameters
    ----------
    center : units.Quantity or EquatorialCoordinates
        The center of the pattern in equatorial coordinates.
    t_interval : units.Quantity
        The sampling interval between output points.
    kwargs : dict, optional
        A list of optional keyword arguments to pass into
        :func:`lissajous_offset`.

    Returns
    -------
    pattern : EquatorialCoordinates
        The equatorial scan pattern sampled at `t_interval`.
    """
    if 'width' in kwargs:
        width = kwargs['width']
    else:
        width = 2 * units.Unit('arcmin')
    if 'height' in kwargs:
        height = kwargs['height']
    else:
        height = 2 * units.Unit('arcmin')
    if 'delta' in kwargs:
        delta = kwargs['delta']
    else:
        delta = np.pi * units.Unit('radian') / 2
    if 'ratio' in kwargs:
        ratio = kwargs['ratio']
    else:
        ratio = np.sqrt(2)
    if 'n_oscillations' in kwargs:
        n_oscillations = kwargs['n_oscillations']
    else:
        n_oscillations = 20
    if 'oscillation_period' in kwargs:
        oscillation_period = kwargs['oscillation_period']
    else:
        oscillation_period = 10 * units.Unit('second')
    if 'constant_speed' in kwargs:
        constant_speed = kwargs['constant_speed']
    else:
        constant_speed = False

    equatorial_offset = lissajous_offset(width, height, t_interval,
                                         ratio=ratio,
                                         delta=delta,
                                         n_oscillations=n_oscillations,
                                         oscillation_period=oscillation_period,
                                         constant_speed=constant_speed)
    equatorial = EquatorialCoordinates(center)
    equatorial.add(equatorial_offset)
    return equatorial
