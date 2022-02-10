# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.coordinate_systems import \
    coordinate_systems_numba_functions as cnf


def test_check_null():
    x = np.zeros((3, 4))
    x[1, 2] = 1
    mask = cnf.check_null(x)
    assert np.allclose(mask, [True, True, False, True])


def test_check_nan():
    x = np.zeros((3, 4))
    x[1, 2] = np.nan
    mask = cnf.check_nan(x)
    assert np.allclose(mask, [False, False, True, False])


def test_check_finite():
    x = np.zeros((3, 4))
    x[1, 2] = np.inf
    mask = cnf.check_finite(x)
    assert np.allclose(mask, [True, True, False, True])


def test_check_infinite():
    x = np.zeros((3, 4))
    x[1, 2] = np.inf
    mask = cnf.check_infinite(x)
    assert np.allclose(mask, [False, False, True, False])


def test_check_value():
    x = np.zeros((3, 4))
    x[:, 1] = -1
    mask = cnf.check_value(-1, x)
    assert np.allclose(mask, [False, True, False, False])


def test_spherical_distance_to():
    # Vincenty
    x = (np.arange(6) * units.Unit('degree')).to('radian').value
    y = x.copy()
    cos_lat = np.cos(y)
    sin_lat = np.sin(y)
    rx = x.copy()
    r_cos_lat = cos_lat.copy()
    r_sin_lat = sin_lat.copy()

    d = cnf.spherical_distance_to(x, rx, cos_lat, sin_lat,
                                  r_cos_lat, r_sin_lat)
    assert d.shape == (6,) and np.allclose(d, 0)

    # Check single coordinate
    sx = np.atleast_1d(x[1])
    s_cos_lat = np.atleast_1d(cos_lat[1])
    s_sin_lat = np.atleast_1d(sin_lat[1])

    d = cnf.spherical_distance_to(
        sx, rx, s_cos_lat, s_sin_lat, r_cos_lat, r_sin_lat)
    expected = np.asarray([0.02468206, 0, 0.0246783,
                           0.04934907, 0.07400856, 0.09865299])
    assert np.allclose(d, expected, atol=1e-5)

    # Check single reference
    d = cnf.spherical_distance_to(
        rx, sx, r_cos_lat, r_sin_lat, s_cos_lat, s_sin_lat)
    assert np.allclose(d, expected, atol=1e-5)

    # rule of cosines
    rsx = np.asarray([np.pi / 2])
    d = cnf.spherical_distance_to(
        x, rsx, cos_lat, sin_lat, s_cos_lat, s_sin_lat)
    assert np.allclose(d, [1.57079633, 1.55304372, 1.53530687,
                           1.51760158, 1.49994369, 1.48234911], atol=1e-5)

    # Check dimensionality is irrelevant
    d = cnf.spherical_distance_to(x.reshape(2, 3), rx.reshape(2, 3),
                                  cos_lat.reshape(2, 3), sin_lat.reshape(2, 3),
                                  r_cos_lat.reshape(2, 3),
                                  r_sin_lat.reshape(2, 3))
    assert d.shape == (2, 3) and np.allclose(d, 0)


def test_spherical_pole_transform():
    phi0 = 0.0
    px = 0.0
    p_cos_lat = 0.0
    p_sin_lat = 1.0

    x = (np.arange(6) * units.Unit('degree')).to('radian').value
    y = x.copy()
    cos_lat = np.cos(y)
    sin_lat = np.sin(y)

    # Test multiple coordinates/poles
    c = cnf.spherical_pole_transform(
        x=x,
        px=np.full(x.shape, px),
        cos_lat=cos_lat,
        sin_lat=sin_lat,
        p_cos_lat=np.full(x.shape, p_cos_lat),
        p_sin_lat=np.full(x.shape, p_sin_lat),
        phi0=phi0,
        reverse=False)

    assert np.allclose(
        c,
        [[3.1415926, 3.1590459, 3.1764992, 3.1939525, 3.2114058, 3.2288591],
         [0, 0.0174532, 0.0349065, 0.0523598, 0.0698131, 0.0872664]],
        atol=1e-4)
    expected_forward = c.copy()

    # Test multiple coordinates/poles (reverse)
    c = cnf.spherical_pole_transform(
        x=x,
        px=np.full(x.shape, px),
        cos_lat=cos_lat,
        sin_lat=sin_lat,
        p_cos_lat=np.full(x.shape, p_cos_lat),
        p_sin_lat=np.full(x.shape, p_sin_lat),
        phi0=phi0,
        reverse=True)
    expected_reverse = c.copy()

    assert np.allclose(
        c,
        [[3.14159265, 3.15904595, 3.17649924, 3.19395253, 3.21140582,
          3.22885912],
         [0, 0.01745329, 0.03490659, 0.05235988, 0.06981317, 0.08726646]],
        atol=1e-4)

    # Test single pole
    for (reverse, expected) in zip([False, True],
                                   [expected_forward, expected_reverse]):

        assert np.allclose(cnf.spherical_pole_transform(
            x=x,
            px=np.full(1, px),
            cos_lat=cos_lat,
            sin_lat=sin_lat,
            p_cos_lat=np.full(1, p_cos_lat),
            p_sin_lat=np.full(1, p_sin_lat),
            phi0=phi0,
            reverse=reverse), expected)

    # Test single coordinate
    c = cnf.spherical_pole_transform(
        x=x[2:3],
        px=np.full(x.shape, px),
        cos_lat=cos_lat[2:3],
        sin_lat=sin_lat[2:3],
        p_cos_lat=np.full(x.shape, p_cos_lat),
        p_sin_lat=np.full(x.shape, p_sin_lat),
        phi0=phi0,
        reverse=True)
    assert c.shape == (2, 6)
    assert np.allclose(c[0], 3.17649924, atol=1e-4)
    assert np.allclose(c[1], 0.03490659, atol=1e-4)
