# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)
from sofia_redux.scan.simulation.scan_patterns.lissajous import (
    lissajous_offset, lissajous_pattern_equatorial)

arcsec = units.Unit('arcsec')
second = units.Unit('second')
degree = units.Unit('degree')


def test_lissajous_offset():
    width = 20 * arcsec
    height = 10 * arcsec
    t_interval = 1 * second
    pattern = lissajous_offset(width, height, t_interval,
                               delta=90 * degree,
                               n_oscillations=1,
                               oscillation_period=10 * second,
                               constant_speed=True)
    assert np.allclose(
        pattern.x,
        [10, 6.043051, 1.797935, -2.860935, -8.120974,
         -9.335062, -5.969955, -0.956728, 3.974756, 8.09017] * arcsec,
        atol=1e-6)
    assert np.allclose(
        pattern.y,
        [5, 1.442398, -1.739561, -4.317637, -4.527777,
         0.142394, 3.787066, 4.438124, 2.665004, -0.713593] * arcsec,
        atol=1e-6)


def test_lissajous_pattern_equatorial():
    center = EquatorialCoordinates([10, 20])
    t = 1 * second
    pattern = lissajous_pattern_equatorial(center, t)
    assert np.allclose(
        pattern.x[:5],
        [-9.983333, -9.986516, -9.99485, -10.00515, -10.013484] * degree,
        atol=1e-6)
    assert np.allclose(
        pattern.y[:5],
        [20.016667, 20.010509, 19.996585, 19.985185, 19.984733] * degree,
        atol=1e-6)
    pattern = lissajous_pattern_equatorial(
        center, t,
        width=10 * arcsec,
        height=15 * arcsec,
        delta=90 * degree,
        ratio=np.sqrt(2),
        n_oscillations=1,
        oscillation_period=10 * second,
        constant_speed=True
    )
    assert np.allclose(
        pattern.x,
        [-9.998611, -9.999079, -9.999589, -10.000292, -10.0012,
         -10.001332, -10.00113, -10.000264, -9.999389, -9.998876] * degree,
        atol=1e-6)
    assert np.allclose(
        pattern.y,
        [20.002083, 20.000805, 19.999543, 19.998376, 19.998483,
         19.999823, 20.00117, 20.001951, 20.000965, 19.999703] * degree,
        atol=1e-6)
