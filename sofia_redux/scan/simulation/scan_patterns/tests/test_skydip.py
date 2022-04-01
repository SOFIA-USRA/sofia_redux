# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.configuration.dates import DateRange
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)
from sofia_redux.scan.coordinate_systems.geodetic_coordinates import (
    GeodeticCoordinates)
from sofia_redux.scan.simulation.scan_patterns.skydip import (
    skydip_pattern_equatorial)
from sofia_redux.scan.utilities.utils import safe_sidereal_time

arcsec = units.Unit('arcsec')
second = units.Unit('second')
degree = units.Unit('degree')


def test_skydip_pattern_equatorial():
    center = EquatorialCoordinates([10, 20])
    site = GeodeticCoordinates([25, 30])
    date_obs = '2022-03-17T15:12:05.068'
    t_interval = 5 * second
    pattern = skydip_pattern_equatorial(center, t_interval, site, date_obs)
    assert np.allclose(
        pattern.x,
        [-3.395466, -12.127039, -21.260456, -30.811666, -40.749302,
         -50.987689] * degree, atol=1e-6)
    assert np.allclose(
        pattern.y,
        [17.190923, 20.826633, 24.002192, 26.61452, 28.565875,
         29.773854] * degree, atol=1e-6)
    t = DateRange.to_time(date_obs) + (np.arange(pattern.size) * t_interval)
    lst = safe_sidereal_time(t, 'mean', longitude=site.longitude)
    horizontal = pattern.to_horizontal(site, lst)
    assert np.allclose(horizontal.az[1:], horizontal.az[0])
    assert np.allclose(horizontal.el, [30, 39, 48, 57, 66, 75] * degree)

    pattern = skydip_pattern_equatorial(center, t_interval, site, date_obs,
                                        scan_time=30,
                                        start_elevation=20,
                                        end_elevation=50)
    horizontal = pattern.to_horizontal(site, lst)
    assert np.allclose(horizontal.el, [20, 26, 32, 38, 44, 50] * degree)

    with pytest.raises(ValueError) as err:
        _ = skydip_pattern_equatorial(center, t_interval, None, date_obs)
    assert 'Site coordinates must be' in str(err.value)
