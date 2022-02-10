# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
from pathlib import Path

from astropy.units import Unit
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
import numpy as np
import pytest

from sofia_redux.spectroscopy.earthvelocity import (
    cartesian_helio, cartesian_lsr, parse_inputs, earthvelocity)


def test_cartesian_lst():
    assert np.allclose(cartesian_lsr(definition='kinematic').xyz.value,
                       [0.28995937, -17.31726701, 10.00140924])
    assert np.allclose(cartesian_lsr(definition='dynamical').xyz.value,
                       [9, 12, 7])
    with pytest.raises(ValueError):
        cartesian_lsr(definition='_does_not_exist_')


def test_cartesian_helio():
    base_time = Time('2000-01-01T12:00:00')
    assert np.allclose(
        cartesian_helio(base_time, center='barycentric').xyz.value,
        [-0.01720221, -0.00290513, -0.00125952])
    assert np.allclose(
        cartesian_helio(base_time, center='heliocentric').xyz.value,
        [-0.01720758, -0.00289837, -0.00125648])
    assert np.allclose(
        cartesian_helio(base_time, location=EarthLocation(45., 45.),
                        center='barycentric').xyz.value,
        [-0.01709434, -0.0027484, -0.00125951])
    with pytest.raises(ValueError):
        cartesian_helio(base_time, center='_does_not_exist')


def test_parse_inputs():
    result = parse_inputs(1, 1, time='2000-01-01T00:00:00',
                          lat=45, lon=45, height=100)
    assert np.isclose(result.ra.value, 15)
    assert np.isclose(result.dec.value, 1)
    assert result.location is not None
    assert result.obstime is not None


def test_earthvelocity():
    result = earthvelocity(
        1, 1, Time('J2000') + np.arange(2) * Unit('s'),
        time_format=None, time_scale=None)
    assert np.allclose(result['vhelio'].value,
                       [-30.1112252, -30.11122563])
    assert np.allclose(result['vsun'].value,
                       [-4.02676891, -4.02676891])
    assert np.allclose(result['vlsr'].value,
                       [-34.13799411, -34.13799454])
    result = earthvelocity(
        1, 1, Time('J2000') + np.arange(2) * Unit('s'),
        time_format=None, time_scale=None,
        lat=45, lon=45, height=100)
    assert np.allclose(result['vhelio'].value,
                       [-29.85959329, -29.85960927])
    assert np.allclose(result['vsun'].value,
                       [-4.02676891, -4.02676891])
    assert np.allclose(result['vlsr'].value,
                       [-33.8863622, -33.88637819])


def test_offline(mocker, capsys):
    # borrowing some test machinery from astropy.utils.iers
    # to invoke offline errors
    from astropy.utils.iers import iers
    from astropy.utils.data import get_pkg_data_filename
    ame = 30.0
    t = Time.now() + TimeDelta(10, format='jd') * np.arange(2)
    iers_a_file_1 = get_pkg_data_filename(
        os.path.join('data', 'finals2000A-2016-02-30-test'),
        package='astropy.utils.iers.tests')
    iers_a_url_1 = Path(iers_a_file_1).as_uri()

    # standard result
    expected = earthvelocity(1, 1, t,
                             time_format=None, time_scale=None,
                             lat=45, lon=45, height=100)

    # offline, data aged out error
    with iers.conf.set_temp('iers_auto_url', iers_a_url_1):
        with iers.conf.set_temp('iers_auto_url_mirror', iers_a_url_1):
            with iers.conf.set_temp('auto_max_age', ame):
                # this will raise and handle a ValueError
                result = earthvelocity(1, 1, t,
                                       time_format=None, time_scale=None,
                                       lat=45, lon=45, height=100)
    capt = capsys.readouterr()
    assert 'attempting offline calculation' in capt.err
    assert 'value may not be accurate' in capt.err

    # result is returned
    assert np.allclose(result['vhelio'].value, expected['vhelio'].value)
    assert np.allclose(result['vsun'].value, expected['vsun'].value)
    assert np.allclose(result['vlsr'].value, expected['vlsr'].value)
