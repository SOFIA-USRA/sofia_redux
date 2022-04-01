# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.equatorial_coordinates import (
    EquatorialCoordinates)
from sofia_redux.scan.simulation.scan_patterns.daisy import (
    daisy_pattern_equatorial, daisy_pattern_offset)

arcsec = units.Unit('arcsec')
second = units.Unit('second')
degree = units.Unit('degree')


def test_daisy_pattern_offset():
    radius = 10 * arcsec
    radial_period = 60 * second
    t_interval = 20 * second
    pattern = daisy_pattern_offset(radius, radial_period, t_interval,
                                   n_oscillations=3)
    assert isinstance(pattern, Coordinate2D)
    assert np.allclose(pattern.x,
                       [0, 8.03123989, -4.30797964,
                        0, -6.28852125, 8.62407022,
                        0, -2.79734345, -2.86977944] * arcsec, atol=1e-6)
    assert np.allclose(pattern.y,
                       [0, 3.240245, -7.512743,
                        0, 5.954368, -0.79083,
                        0, -8.196028, 8.170946] * arcsec, atol=1e-6)
    # Radially equal spaced points due to radial/rotational phase
    assert np.allclose(pattern.length,
                       [0, 8.660254, 8.660254,
                        0, 8.660254, 8.660254,
                        0, 8.660254, 8.660254] * arcsec, atol=1e-6)

    pattern = daisy_pattern_offset(radius, radial_period, t_interval,
                                   n_oscillations=3, constant_speed=True)
    assert np.allclose(pattern.x,
                       [0, 5.852402, -2.855421,
                        -0.69953, -2.777535, 7.746324,
                        -0.622349, -2.818661, -2.869779] * arcsec, atol=1e-6)
    assert np.allclose(pattern.y,
                       [0, 1.341501, -6.246914,
                        0.662359, 4.366294, -0.393812,
                        -1.82344, -3.379275, 8.170946] * arcsec, atol=1e-6)
    # check no longer spaced equally along radius
    assert np.allclose(pattern.length,
                       [0, 6.004185, 6.868578,
                        0.96336, 5.174865, 7.756328,
                        1.92672, 4.400494, 8.660254] * arcsec, atol=1e-6)


def test_daisy_pattern_equatorial():
    t_interval = 1 * second
    center = EquatorialCoordinates([10, 20])
    pattern = daisy_pattern_equatorial(center, t_interval)
    assert isinstance(pattern, EquatorialCoordinates)
    assert pattern.size == 110
    assert np.allclose(
        pattern.x[:5],
        [-10, -9.971024, -9.9866, -10.006775, -9.998514] * degree, atol=1e-6)
    assert np.allclose(
        pattern.y[:5],
        [20, 20.01286, 20.014294, 19.981616, 19.968333] * degree, atol=1e-6)
    pattern = daisy_pattern_equatorial(center, t_interval,
                                       radius=10 * arcsec,
                                       radial_period=60 * second,
                                       n_oscillations=10,
                                       constant_speed=True)
    assert pattern.size == 600
    assert np.allclose(
        pattern.x[:5],
        [-10, -9.999923, -9.999853, -9.999791, -9.99974] * degree, atol=1e-6)
    assert np.allclose(
        pattern.y[:5],
        [20, 20.000189, 20.000381, 20.000576, 20.000774] * degree, atol=1e-6)
