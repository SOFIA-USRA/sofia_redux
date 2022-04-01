# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import (
    GeodeticCoordinates)
from sofia_redux.scan.configuration.dates import DateRange
from sofia_redux.scan.utilities.utils import safe_sidereal_time


__all__ = ['skydip_pattern_equatorial']


def skydip_pattern_equatorial(center, t_interval, site, date_obs, **kwargs):
    """
    Create a skydip observation.

    Parameters
    ----------
    center : units.Quantity or EquatorialCoordinates
        The center of the pattern in equatorial coordinates.  Only the azimuth
        position is calculated and held constant when creating the simulated
        scan pattern.
    t_interval : units.Quantity
        The sampling interval between output points.
    site : GeodeticCoordinates
        The site of the observation.
    date_obs : str or astropy.time.Time
        The date of the start of the observation in ISOT UTC format.

    Returns
    -------
    pattern : EquatorialCoordinates
        The equatorial scan pattern sampled at `t_interval`.
    """
    if not isinstance(site, GeodeticCoordinates):
        raise ValueError(f"Site coordinates must be {GeodeticCoordinates}. "
                         f"Received: {site}.")

    scan_time = kwargs.get('scan_time', 30 * units.Unit('second'))
    if not isinstance(scan_time, units.Quantity):
        scan_time = scan_time * units.Unit('second')
    start_elevation = kwargs.get('start_elevation', 30 * units.Unit('degree'))
    if not isinstance(start_elevation, units.Quantity):
        start_elevation = start_elevation * units.Unit('degree')
    end_elevation = kwargs.get('end_elevation', 75 * units.Unit('degree'))
    if not isinstance(end_elevation, units.Quantity):
        end_elevation = end_elevation * units.Unit('degree')

    date_obs = DateRange.to_time(date_obs)
    n = int(np.ceil((scan_time / t_interval).decompose().value))
    t = t_interval * np.arange(n) + date_obs
    lst = safe_sidereal_time(t, 'mean', longitude=site.longitude)
    equatorial = EquatorialCoordinates(center)
    horizontal = equatorial.to_horizontal(site, lst)
    azimuth = np.full(n, horizontal.az[0].value) * horizontal.unit
    elevation = np.linspace(start_elevation, end_elevation, n)
    horizontal.az = azimuth
    horizontal.el = elevation

    equatorial = horizontal.to_equatorial(site, lst)
    return equatorial
