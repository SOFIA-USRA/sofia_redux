# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from sofia_redux.scan.coordinate_systems.projection.\
    projection_numba_functions import (
        equal_angles, asin, acos, asin_array, acos_array, spherical_project,
        spherical_deproject, spherical_project_array,
        spherical_deproject_array, calculate_celestial_pole,
        calculate_celestial_pole_array)


def test_equal_angles():
    ra = np.deg2rad(90)
    circle = 2 * np.pi
    assert equal_angles(ra, ra)
    assert equal_angles(ra, circle + ra)
    assert equal_angles(ra, ra - circle)
    assert equal_angles(ra, ra + 1e-13)
    assert not equal_angles(ra, ra + 1e-11)


def test_set_asin():
    assert asin(0) == 0
    assert np.isclose(np.rad2deg(asin(np.sqrt(3) / 2)), 60)
    assert np.isclose(asin(-1), -np.pi / 2)
    assert np.isclose(asin(-2), -np.pi / 2)
    assert np.isclose(asin(1), np.pi / 2)
    assert np.isclose(asin(2), np.pi / 2)


def test_asin_array():
    p2 = np.pi / 2
    x = np.asarray([np.sqrt(3) / 2, -1, -2, 1, 2])
    expected = np.asarray([np.deg2rad(60), -p2, -p2, p2, p2])
    assert np.allclose(asin_array(x), expected)


def test_acos():
    assert acos(0) == np.pi / 2
    assert np.isclose(np.rad2deg(acos(np.sqrt(3) / 2)), 30)
    assert np.isclose(acos(-1), np.pi)
    assert np.isclose(acos(-2), np.pi)
    assert np.isclose(acos(1), 0)
    assert np.isclose(acos(2), 0)


def test_acos_array():
    x = np.asarray([0, np.sqrt(3) / 2, -1, -2, 1, 2])
    p2 = np.pi / 2
    expected = np.asarray([p2, np.deg2rad(30), np.pi, np.pi, 0, 0])
    assert np.allclose(acos_array(x), expected)


def test_spherical_project():
    # Test right angle pole transforms
    x = np.deg2rad(30)
    y = np.deg2rad(40)
    lon_cp = 0.0
    lat_cp = np.deg2rad(90)
    lon_np = np.deg2rad(10)
    cos_lat = np.cos(y)
    sin_lat = np.sin(y)
    cos_lat_cp = np.cos(lat_cp)
    sin_lat_cp = np.sin(lat_cp)

    # 90 degree celestial pole latitude
    theta, phi = spherical_project(
        x=x, y=y, cos_lat=cos_lat, sin_lat=sin_lat,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    x_cp = np.pi + lon_np + x - lon_cp
    y_cp = y
    assert np.isclose(x_cp, phi)
    assert np.isclose(y_cp, theta)

    # -90 degree celestial pole latitude
    lat_cp = -lat_cp
    cos_lat_cp = np.cos(lat_cp)
    sin_lat_cp = np.sin(lat_cp)
    theta, phi = spherical_project(
        x=x, y=y, cos_lat=cos_lat, sin_lat=sin_lat,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    x_cp = lon_np + lon_cp - x
    y_cp = -y
    assert np.isclose(x_cp, phi)
    assert np.isclose(y_cp, theta)

    # Not a right angle pole
    lon_cp = np.deg2rad(20)
    lat_cp = np.deg2rad(25)
    cos_lat_cp = np.cos(lat_cp)
    sin_lat_cp = np.sin(lat_cp)
    theta, phi = spherical_project(
        x=x, y=y, cos_lat=cos_lat, sin_lat=sin_lat,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    d = x - lon_cp
    a = -np.cos(y) * np.sin(d)
    b = np.sin(y) * np.cos(lat_cp)
    b -= np.cos(y) * np.sin(lat_cp) * np.cos(d)
    x_cp = lon_np + np.arctan2(a, b)
    y_cp = np.arcsin(
        (np.sin(y) * np.sin(lat_cp))
        + (np.cos(y) * np.cos(lat_cp) * np.cos(d))
    )
    assert np.isclose(x_cp, phi)
    assert np.isclose(y_cp, theta)


def test_spherical_deproject():
    # Test right angle pole transforms
    x_cp = np.deg2rad(25)
    y_cp = np.deg2rad(35)
    lon_cp = 0.0
    lat_cp = np.deg2rad(90)
    lon_np = np.deg2rad(10)
    cos_lat_cp = np.cos(lat_cp)
    sin_lat_cp = np.sin(lat_cp)

    # 90 degree celestial pole latitude
    cx, cy = spherical_deproject(
        phi=x_cp, theta=y_cp,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    x = lon_cp + x_cp - lon_np - np.pi
    y = y_cp
    assert np.allclose([cx, cy], [x, y])

    # -90 degree celestial pole latitude
    lat_cp = -lat_cp
    cos_lat_cp = np.cos(lat_cp)
    sin_lat_cp = np.sin(lat_cp)
    cx, cy = spherical_deproject(
        phi=x_cp, theta=y_cp,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    x = lon_cp + lon_np - x_cp
    y = -y_cp
    assert np.allclose([cx, cy], [x, y])

    # Not a right angle pole
    lon_cp = np.deg2rad(20)
    lat_cp = np.deg2rad(25)
    cos_lat_cp = np.cos(lat_cp)
    sin_lat_cp = np.sin(lat_cp)
    cx, cy = spherical_deproject(
        phi=x_cp, theta=y_cp,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    d_cp = x_cp - lon_np
    c = -np.cos(y_cp) * np.sin(d_cp)
    d = (np.sin(y_cp) * np.cos(lat_cp)) - (
        np.cos(y_cp) * np.sin(lat_cp) * np.cos(d_cp))
    x = lon_cp + np.arctan2(c, d)

    e = np.sin(y_cp) * np.sin(lat_cp)
    e += np.cos(y_cp) * np.cos(lat_cp) * np.cos(d_cp)
    y = np.arcsin(e)
    assert np.allclose([cx, cy], [x, y])


def test_spherical_project_array():
    # For ease of testing, just use a north pole
    lon_cp = 0.0
    lat_cp = np.deg2rad(90.0)
    lon_np = 0.0
    x = np.arange(10, dtype=float)
    y = x + 10
    cos_lat = np.cos(y)
    sin_lat = np.sin(y)
    cos_lat_cp = 0.0
    sin_lat_cp = 1.0

    # Multiple coordinates, single poles
    y_cp, x_cp = spherical_project_array(
        x=x, y=y, cos_lat=cos_lat, sin_lat=sin_lat,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    assert np.allclose(y_cp, y)
    assert np.allclose(x_cp, np.pi + lon_np + x - lon_cp)

    # Single coordinate, multiple celestial poles
    x = 1.0
    y = 2.0
    cos_lat = np.cos(y)
    sin_lat = np.sin(y)
    lon_cp = np.full(10, lon_cp)
    lat_cp = np.full(10, lat_cp)
    y_cp, x_cp = spherical_project_array(
        x=x, y=y, cos_lat=cos_lat, sin_lat=sin_lat,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)
    assert np.allclose(y_cp, np.full(10, 2))
    assert np.allclose(x_cp, np.full(10, 1 + np.pi))

    # Multiple coordinates and poles
    x = np.arange(10, dtype=float)
    y = x + 10
    cos_lat = np.cos(y)
    sin_lat = np.sin(y)
    y_cp, x_cp = spherical_project_array(
        x=x, y=y, cos_lat=cos_lat, sin_lat=sin_lat,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)
    assert np.allclose(y_cp, y)
    assert np.allclose(x_cp, x + np.pi)


def test_spherical_deproject_array():
    # For ease of testing, just use a north pole
    x_cp = np.arange(10, dtype=float)
    y_cp = x_cp + 10
    lon_cp = 0.0
    lat_cp = np.deg2rad(90.0)
    lon_np = 0.0
    cos_lat_cp = 0.0
    sin_lat_cp = 1.0

    # Multiple coordinates, single poles
    x, y = spherical_deproject_array(
        phi=x_cp, theta=y_cp,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)

    assert np.allclose(x, x_cp - np.pi)
    assert np.allclose(y, y_cp)

    # Single coordinate, multiple celestial poles
    x_cp, y_cp = 1.0, 2.0
    lon_cp = np.full(10, lon_cp)
    lat_cp = np.full(10, lat_cp)
    x, y = spherical_deproject_array(
        phi=x_cp, theta=y_cp,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)
    assert np.allclose(x, np.full(10, 1 - np.pi))
    assert np.allclose(y, np.full(10, 2))

    # Multiple coordinates and poles
    x_cp = np.arange(10, dtype=float)
    y_cp = x_cp + 10
    x, y = spherical_deproject_array(
        phi=x_cp, theta=y_cp,
        celestial_pole_x=lon_cp, celestial_pole_y=lat_cp,
        celestial_cos_lat=cos_lat_cp, celestial_sin_lat=sin_lat_cp,
        native_pole_x=lon_np)
    assert np.allclose(x, x_cp - np.pi)
    assert np.allclose(y, y_cp)


def test_calculate_celestial_pole():
    native_reference_x = 0.0
    native_reference_y = 0.0
    native_reference_cos_lat = np.cos(native_reference_y)
    native_reference_sin_lat = np.sin(native_reference_y)
    native_pole_x = 0.0
    native_pole_y = np.deg2rad(90)
    reference_x = np.deg2rad(30)
    reference_y = np.deg2rad(60)
    reference_cos_lat = np.cos(reference_y)
    reference_sin_lat = np.sin(reference_y)
    cx, cy = calculate_celestial_pole(
        native_reference_x=native_reference_x,
        native_reference_cos_lat=native_reference_cos_lat,
        native_reference_sin_lat=native_reference_sin_lat,
        reference_x=reference_x,
        reference_y=reference_y,
        reference_cos_lat=reference_cos_lat,
        reference_sin_lat=reference_sin_lat,
        native_pole_x=native_pole_x,
        native_pole_y=native_pole_y,
        select_solution=1
    )
    assert np.isclose(np.rad2deg(cx), 30)
    assert np.isclose(np.rad2deg(cy), -30)

    cx, cy = calculate_celestial_pole(
        native_reference_x=native_reference_x,
        native_reference_cos_lat=native_reference_cos_lat,
        native_reference_sin_lat=native_reference_sin_lat,
        reference_x=reference_x,
        reference_y=reference_y,
        reference_cos_lat=reference_cos_lat,
        reference_sin_lat=reference_sin_lat,
        native_pole_x=native_pole_x,
        native_pole_y=native_pole_y,
        select_solution=-1
    )
    assert np.isclose(np.rad2deg(cx), -150)
    assert np.isclose(np.rad2deg(cy), 30)

    cx, cy = calculate_celestial_pole(
        native_reference_x=native_reference_x,
        native_reference_cos_lat=native_reference_cos_lat,
        native_reference_sin_lat=native_reference_sin_lat,
        reference_x=reference_x,
        reference_y=reference_y,
        reference_cos_lat=reference_cos_lat,
        reference_sin_lat=reference_sin_lat,
        native_pole_x=native_pole_x,
        native_pole_y=native_pole_y,
        select_solution=-0)

    assert np.isclose(np.rad2deg(cx), -150)
    assert np.isclose(np.rad2deg(cy), 30)


def test_calculate_celestial_pole_array():

    native_reference_x = np.zeros(3)
    native_reference_y = np.zeros(3)
    native_reference_cos_lat = np.cos(native_reference_y)
    native_reference_sin_lat = np.sin(native_reference_y)
    native_pole_x = np.zeros(3)
    native_pole_y = np.deg2rad(np.full(3, 90))
    reference_x = np.deg2rad(np.full(3, 30))
    reference_y = np.deg2rad(np.full(3, 60))
    reference_cos_lat = np.cos(reference_y)
    reference_sin_lat = np.sin(reference_y)

    # All arrays
    cx, cy = calculate_celestial_pole_array(
        native_reference_x=native_reference_x,
        native_reference_cos_lat=native_reference_cos_lat,
        native_reference_sin_lat=native_reference_sin_lat,
        reference_x=reference_x,
        reference_y=reference_y,
        reference_cos_lat=reference_cos_lat,
        reference_sin_lat=reference_sin_lat,
        native_pole_x=native_pole_x,
        native_pole_y=native_pole_y,
        select_solution=1
    )
    assert cx.size == 3 and cy.size == 3
    assert np.allclose(np.rad2deg(cx), 30)
    assert np.allclose(np.rad2deg(cy), -30)

    # native reference_array
    cx, cy = calculate_celestial_pole_array(
        native_reference_x=native_reference_x,
        native_reference_cos_lat=native_reference_cos_lat,
        native_reference_sin_lat=native_reference_sin_lat,
        reference_x=reference_x[0],
        reference_y=reference_y[0],
        reference_cos_lat=reference_cos_lat[0],
        reference_sin_lat=reference_sin_lat[0],
        native_pole_x=native_pole_x[0],
        native_pole_y=native_pole_y[0],
        select_solution=-1
    )
    assert cx.size == 3 and cy.size == 3
    assert np.allclose(np.rad2deg(cx), -150)
    assert np.allclose(np.rad2deg(cy), 30)

    # native reference array
    cx, cy = calculate_celestial_pole_array(
        native_reference_x=native_reference_x[0],
        native_reference_cos_lat=native_reference_cos_lat[0],
        native_reference_sin_lat=native_reference_sin_lat[0],
        reference_x=reference_x,
        reference_y=reference_y,
        reference_cos_lat=reference_cos_lat,
        reference_sin_lat=reference_sin_lat,
        native_pole_x=native_pole_x[0],
        native_pole_y=native_pole_y[0],
        select_solution=0
    )

    assert cx.size == 3 and cy.size == 3
    assert np.allclose(np.rad2deg(cx), -150)
    assert np.allclose(np.rad2deg(cy), 30)

    # native pole array
    cx, cy = calculate_celestial_pole_array(
        native_reference_x=native_reference_x[0],
        native_reference_cos_lat=native_reference_cos_lat[0],
        native_reference_sin_lat=native_reference_sin_lat[0],
        reference_x=reference_x[0],
        reference_y=reference_y[0],
        reference_cos_lat=reference_cos_lat[0],
        reference_sin_lat=reference_sin_lat[0],
        native_pole_x=native_pole_x,
        native_pole_y=native_pole_y,
        select_solution=1
    )

    assert cx.size == 3 and cy.size == 3
    assert np.allclose(np.rad2deg(cx), 30)
    assert np.allclose(np.rad2deg(cy), -30)
