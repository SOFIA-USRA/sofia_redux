# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['check_null', 'check_nan', 'check_finite',
           'check_infinite', 'check_value', 'spherical_distance_to',
           'spherical_pole_transform']


@nb.njit(cache=True, nogil=False, parallel=False)
def check_null(coordinates):  # pragma: no cover
    """
    Check if coordinates are zeroed.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        Coordinates of shape (n_dimensions, N).

    Returns
    -------
    numpy.ndarray (bool)
    """
    n_dimensions, n = coordinates.shape
    result = np.empty(n, dtype=nb.b1)
    for i in range(n):
        for dimension in range(n_dimensions):
            if coordinates[dimension, i] != 0:
                result[i] = False
                break
        else:
            result[i] = True
    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def check_nan(coordinates):  # pragma: no cover
    """
    Check if coordinates are NaN.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        Coordinates of shape (n_dimensions, N).

    Returns
    -------
    numpy.ndarray (bool)
    """
    n_dimensions, n = coordinates.shape
    result = np.empty(n, dtype=nb.b1)
    for i in range(n):
        for dimension in range(n_dimensions):
            if np.isnan(coordinates[dimension, i]):
                result[i] = True
                break
        else:
            result[i] = False
    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def check_finite(coordinates):  # pragma: no cover
    """
    Check if coordinates are finite.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        Coordinates of shape (n_dimensions, N).

    Returns
    -------
    numpy.ndarray (bool)
    """
    n_dimensions, n = coordinates.shape
    result = np.empty(n, dtype=nb.b1)
    for i in range(n):
        for dimension in range(n_dimensions):
            if not np.isfinite(coordinates[dimension, i]):
                result[i] = False
                break
        else:
            result[i] = True
    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def check_infinite(coordinates):  # pragma: no cover
    """
    Check if coordinates are infinite.

    Parameters
    ----------
    coordinates : numpy.ndarray (float)
        Coordinates of shape (n_dimensions, N).

    Returns
    -------
    numpy.ndarray (bool)
    """
    n_dimensions, n = coordinates.shape
    result = np.empty(n, dtype=nb.b1)
    for i in range(n):
        for dimension in range(n_dimensions):
            if np.isinf(coordinates[dimension, i]):
                result[i] = True
                break
        else:
            result[i] = False
    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def check_value(value, coordinates):  # pragma: no cover
    """
    Check if coordinates are equal to a given value in all dimensions.

    Parameters
    ----------
    value : int or float
        The value
    coordinates : numpy.ndarray (float)
        Coordinates of shape (n_dimensions, N).

    Returns
    -------
    numpy.ndarray (bool)
    """
    n_dimensions, n = coordinates.shape
    result = np.empty(n, dtype=nb.b1)
    for i in range(n):
        for dimension in range(n_dimensions):
            if coordinates[dimension, i] != value:
                result[i] = False
                break
        else:
            result[i] = True
    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def spherical_distance_to(x, rx, cos_lat, sin_lat, r_cos_lat, r_sin_lat
                          ):  # pragma: no cover
    r"""
    Return the angular distance in radians between spherical coordinate sets.

    Calculates the distance between two spherical sets of coordinates using
    either the law of cosines or Vincenty's formulae.  First we calculate
    c as::

      c = sin(y) * sin(ry) + cos(y) * phi

    where::

      phi = cos(ry) * cos(rx - x)

    and x, rx are the longitudinal coordinates or the coordinates and reference
    coordinates respectively, and (y, ry) are the latitudinal coordinates.

    if \|c\| > 0.9 (indicating intermediate distances), the law of cosines
    is used to return an angle (a) of::

      a = acos(c)

    Otherwise, Vincenty's formula is used to return a value of::

      a = atan2(B, c)

    where::

      B = sqrt((cos(ry) * sin(rx - x))^2 + (cos(y) * sin(ry) - sin(y) * phi)^2)

    Parameters
    ----------
    x : numpy.ndarray (float)
        The x-direction spherical coordinates in radians of the coordinate to
        test.  Must either be of shape (1,) or (n,).
    rx : numpy.ndarray (float)
        The x-direction spherical reference coordinate in radians.  Must either
        be of shape (1,) or (n,).
    cos_lat : numpy.ndarray (float)
        The cosine(Latitude) of the spherical coordinate.  Must match the shape
        of `x`.
    sin_lat : numpy.ndarray (float)
        The sine(Latitude) of the spherical coordinate.  Must match the shape
        of `x`.
    r_cos_lat : numpy.ndarray (float)
        The cosine(Latitude) of the spherical reference coordinate.  Must match
        the shape of `rx`.
    r_sin_lat : numpy.ndarray (float)
        The sine(Latitude) of the spherical reference coordinate.  Must match
        the shape of `rx`.

    Returns
    -------
    distance : numpy.ndarray (float)
       The distance in radians between the coordinates and reference
       coordinates.  Will be of shape (1,) or (n,) depending on `x` and `rx`.
    """
    single_coord = x.size == 1
    single_ref = rx.size == 1

    flat_cos_lat = cos_lat.flat
    flat_sin_lat = sin_lat.flat
    flat_r_cos_lat = r_cos_lat.flat
    flat_r_sin_lat = r_sin_lat.flat

    # All these are correct shape
    dl = rx - x
    cos_phi2_cos_dl = r_cos_lat * np.cos(dl)
    c = (sin_lat * r_sin_lat) + (cos_lat * cos_phi2_cos_dl)
    flat_c = c.flat
    flat_phi = cos_phi2_cos_dl.flat
    flat_dl = dl.flat

    result = np.empty_like(c)
    flat_result = result.flat

    if single_coord:
        cl = flat_cos_lat[0]
        sl = flat_sin_lat[0]
    else:
        cl = sl = 0.0

    if single_ref:
        rcl = flat_r_cos_lat[0]
        rsl = flat_r_sin_lat[0]
    else:
        rcl = rsl = 0.0

    for i in range(c.size):
        ci = flat_c[i]
        if -0.9 < ci < 0.9:
            flat_result[i] = np.arccos(ci)
            continue

        if not single_coord:
            cl = flat_cos_lat[i]
            sl = flat_sin_lat[i]
        if not single_ref:
            rcl = flat_r_cos_lat[i]
            rsl = flat_r_sin_lat[i]

        dx = rcl * np.sin(flat_dl[i])
        dy = (cl * rsl) - (sl * flat_phi[i])
        flat_result[i] = np.arctan2(np.hypot(dx, dy), ci)

    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def spherical_pole_transform(x, px, cos_lat, sin_lat, p_cos_lat,
                             p_sin_lat, phi0, reverse=False
                             ):  # pragma: no cover
    """
    Transform spherical coordinates to a new pole.

    The transformation occurs according to::

       xt = arcsin((sin(py) * sin(y)) + (cos(py) * cos(y) * cos(dl)))
       yt = o + arctan2((-sin(y) * cos(py)) + (cos(y) * sin(py) * cos(dl)),
                        (-cos(y) * sin(dl)))

    For when `reverse=False`::

       dl = x - px
       o = pi/2 - phi0

    and when `reverse=True`::

       dl = x + phi0
       o = x + pi/2

    Here, (x, px) refer respectively to the coordinate and pole longitudes,
    while (y, py) refer to latitudes.

    Parameters
    ----------
    x : numpy.ndarray (float)
        The coordinate LON values in radians of shape (1,) or (n,).
    px : numpy.ndarray (float)
        The new pole longitude position in radians of shape (1,) or (n,).
    cos_lat : numpy.ndarray (float)
        The cosine(latitude) values of shape (1,) or (n,).
    sin_lat : numpy.ndarray (float)
        The sine(latitude) values of shape (1,) or (n,).
    p_cos_lat : numpy.ndarray (float)
        The cosine(latitude) values of the pole of shape (1,) or (n,).
    p_sin_lat : numpy.ndarray (float)
        The sine(latitude) values of the pole od shape (1,) or (n,).
    phi0 : float
        The phi0 angle in radians.
    reverse : bool, optional
        If `True`, perform the inverse transform (transform from pole rather
        than to).

    Returns
    -------
    transformed_coordinates : numpy.ndarray (float)
        The transformed coordinates in radians of shape (2, n or 1).
    """
    n = max(x.size, px.size)
    single_pole = px.size == 1
    single_coordinate = x.size == 1

    flat_x = x.flat
    flat_px = px.flat
    flat_cos_lat = cos_lat.flat
    flat_sin_lat = sin_lat.flat
    flat_p_cos_lat = p_cos_lat.flat
    flat_p_sin_lat = p_sin_lat.flat

    xi = flat_x[0]
    cli = flat_cos_lat[0]
    sli = flat_sin_lat[0]
    p_xi = flat_px[0]
    p_cli = flat_p_cos_lat[0]
    p_sli = flat_p_sin_lat[0]

    right_angle = np.pi / 2.0

    result = np.empty((2, n), dtype=nb.float64)
    for i in range(n):
        if single_coordinate:
            lon = xi
            cl = cli
            sl = sli
        else:
            lon = flat_x[i]
            cl = flat_cos_lat[i]
            sl = flat_sin_lat[i]

        if single_pole:
            p_lon = p_xi
            p_cl = p_cli
            p_sl = p_sli
        else:
            p_lon = flat_px[i]
            p_cl = flat_p_cos_lat[i]
            p_sl = flat_p_sin_lat[i]

        if reverse:
            dl = lon + phi0
            offset = p_lon + right_angle  # TODO: changed from lon to p_lon
        else:
            dl = lon - p_lon
            offset = right_angle - phi0

        cos_dl, sin_dl = np.cos(dl), np.sin(dl)

        new_lat = np.arcsin((p_sl * sl) + (p_cl * cl * cos_dl))
        new_lon = offset + np.arctan2((-sl * p_cl) + (cl * p_sl * cos_dl),
                                      (-cl * sin_dl))
        result[0, i] = new_lon
        result[1, i] = new_lat
    return result
