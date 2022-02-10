# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

nb.config.THREADING_LAYER = 'threadsafe'
two_pi = 2 * np.pi

__all__ = ['spherical_project_array', 'spherical_project',
           'spherical_deproject_array', 'spherical_deproject',
           'calculate_celestial_pole_array', 'calculate_celestial_pole',
           'equal_angles', 'asin', 'asin_array', 'acos', 'acos_array']


@nb.njit(cache=True, nogil=False, parallel=False)
def equal_angles(a1, a2, angular_accuracy=1e-12):  # pragma: no cover
    """
    Check if two angles are equal within accuracy.

    Parameters
    ----------
    a1 : float
        The first angle in radians.
    a2 : float
        The second angle in radians.
    angular_accuracy : float, optional
        The angular accuracy in radians.

    Returns
    -------
    equal : bool
    """
    return np.abs(np.fmod(a1 - a2, two_pi)) < angular_accuracy


@nb.njit(cache=True, nogil=False, parallel=False)
def asin(value):  # pragma: no cover
    """
    Return the inverse sine of a value.

    Values are clipped between -1 <= x <= 1 before being passed into
    :func:`np.arcsin`.

    Parameters
    ----------
    value : float

    Returns
    -------
    angle : float
        The angle in radians
    """
    if value < -1:
        value = -1.0
    elif value > 1:
        value = 1.0
    return np.arcsin(value)


@nb.njit(cache=True, nogil=False, parallel=False)
def asin_array(values):  # pragma: no cover
    """
    Return the inverse sine for an array of values.

    This is a wrapper that passes an array of values into :func:`asin`.

    Parameters
    ----------
    values : numpy.ndarray (float)

    Returns
    -------
    angles : numpy.ndarray (float)
        The angles in radians.
    """
    result = np.empty_like(values, dtype=nb.float64)
    flat_result = result.flat
    flat_values = values.flat
    for i in range(values.size):
        flat_result[i] = asin(flat_values[i])
    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def acos(value):  # pragma: no cover
    """
    Return the inverse cosine of an angle.

    Parameters
    ----------
    value : float
        The angle in radians.

    Returns
    -------
    value : float
    """
    if value < -1:
        value = -1.0
    elif value > 1:
        value = 1.0
    return np.arccos(value)


@nb.njit(cache=True, nogil=False, parallel=False)
def acos_array(values):  # pragma: no cover
    """
    Return the inverse cosine for an array of values

        This is a wrapper that passes an array of values into :func:`acos`.

    Parameters
    ----------
    values : numpy.ndarray (float)

    Returns
    -------
    angles : numpy.ndarray (float)
        The angles in radians.
    """
    result = np.empty_like(values, dtype=nb.float64)
    flat_result = result.flat
    flat_values = values.flat
    for i in range(values.size):
        flat_result[i] = acos(flat_values[i])
    return result


@nb.njit(cache=True, nogil=False, parallel=False)
def spherical_project(x, y, cos_lat, sin_lat,
                      celestial_pole_x, celestial_pole_y,
                      celestial_cos_lat, celestial_sin_lat, native_pole_x
                      ):  # pragma: no cover
    """
    Convert a single coordinate form a native pole to a celestial pole.

    The following conversions are used depending on the value of the celestial
    pole native latitude (lat_cp) to convert the coordinates (x, y) to
    coordinates about the celestial pole (x_cp, y_cp).  Here _cp denotes
    "celestial pole", _np denotes "native pole", and any numbers are in
    degrees.

    lat_cp = 90:

        x_cp = 180 + lon_np + x - lon_cp
        y_cp = y

    lat_cp = -90:

        x_cp = lon_np + lon_cp - x
        y_cp = -y

    Otherwise:

        d = x - lon_cp
        A = -cos(y)sin(d)
        B = sin(y)cos(lat_cp) - cos(y)sin(lat_cp)cos(d)
        x_cp = lon_np + arctan2(A, B)
        y_cp = arcsin(sin(y)sin(lat_cp) + cos(y)cos(lat_cp)cos(d))

    The reverse operation (celestial to native poles) can be performed using
    :func:`spherical_deproject`.

    Parameters
    ----------
    x : float
        The coordinate native longitude in radians.
    y : float
        The coordinate native latitude in radians.
    cos_lat : float
        The cosine of `y`.
    sin_lat : float
        The sine of `y`.
    celestial_pole_x : float
        The celestial pole native longitude in radians.
    celestial_pole_y : float
        The celestial pole native latitude in radians.
    celestial_cos_lat : float
        The cosine of `celestial_pole_y`.
    celestial_sin_lat : float
        The sine of `celestial_pole_y`.
    native_pole_x : float
        The coordinate's native pole longitude in radians.

    Returns
    -------
    theta, phi : float, float
        The projection of (x, y) about the native pole on the celestial pole,
        with `theta` equivalent to y_cp and `phi` equivalent to x_cp where _cp
        denotes a projection about the celestial pole.
    """
    right_angle = np.pi / 2

    d_lon = x - celestial_pole_x
    if equal_angles(np.abs(celestial_pole_y), right_angle):
        if celestial_pole_y > 0:
            phi = native_pole_x + d_lon + np.pi
            theta = y
        else:
            phi = native_pole_x - d_lon
            theta = -y
    else:
        cos_d_lon = np.cos(d_lon)

        phi = native_pole_x + np.arctan2(
            -cos_lat * np.sin(d_lon),
            (sin_lat * celestial_cos_lat)
            - (cos_lat * celestial_sin_lat * cos_d_lon))

        theta = asin(
            (sin_lat * celestial_sin_lat)
            + (cos_lat * celestial_cos_lat * cos_d_lon))

        phi = np.fmod(phi, two_pi)

    return theta, phi


@nb.njit(cache=True, nogil=False, parallel=False)
def spherical_project_array(x, y, cos_lat, sin_lat,
                            celestial_pole_x, celestial_pole_y,
                            celestial_cos_lat, celestial_sin_lat, native_pole_x
                            ):  # pragma: no cover
    """
    Project multiple coordinates from a native pole onto a celestial pole.

    This function is a wrapper around :func:`spherical_project` to process
    projections when at least one of either the native coordinates, celestial
    pole, or native pole consists of array values.

    Parameters
    ----------
    x : float or numpy.ndarray
        The native longitude coordinates in radians.  If an array is passed in,
        it should be of shape (1,) or (n,).
    y : float or numpy.ndarray
        The native latitude coordinates in radians.  If an array is passed in,
        it should be of shape (1,) or (n,)
    cos_lat : float or numpy.ndarray
        The cosine of `y`.
    sin_lat : float or numpy.ndarray
        The sine of `y`.
    celestial_pole_x : float or numpy.ndarray
        The native longitude of the celestial pole.  If an array is passed in,
        it should be of shape (1,) or (n,).
    celestial_pole_y : float or numpy.ndarray
        The native latitude of the celestial pole.  If an array is passed in,
        it should be of shape (1,) or (n,).
    celestial_cos_lat : float or numpy.ndarray
        The cosine of `celestial_pole_y`.
    celestial_sin_lat : float or numpy.ndarray
        The sine of `celestial_pole_y`.
    native_pole_x : float or numpy.ndarray
        The spherical projection's native pole longitude.  If an array
        is passed in, it should be of shape (1,) or (n,).

    Returns
    -------
    theta, phi : numpy.ndarray, numpy.ndarray
        The projection of (x, y) about the native pole on the celestial pole,
        with `theta` equivalent to y_p and `phi` equivalent to x_p where _p
        denotes a projection about the celestial pole.
    """
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    cos_lat = np.atleast_1d(np.asarray(cos_lat))
    sin_lat = np.atleast_1d(np.asarray(sin_lat))
    celestial_pole_x = np.atleast_1d(np.asarray(celestial_pole_x))
    celestial_pole_y = np.atleast_1d(np.asarray(celestial_pole_y))
    celestial_cos_lat = np.atleast_1d(np.asarray(celestial_cos_lat))
    celestial_sin_lat = np.atleast_1d(np.asarray(celestial_sin_lat))
    native_pole_x = np.atleast_1d(np.asarray(native_pole_x))

    sizes = np.array([x.size, celestial_pole_x.size, native_pole_x.size])
    max_array = np.argmax(sizes)
    if max_array == 0:
        theta = np.empty_like(x, dtype=nb.float64)
        phi = np.empty_like(x, dtype=nb.float64)
        n = x.size
    else:
        theta = np.empty_like(celestial_pole_x, dtype=nb.float64)
        phi = np.empty_like(celestial_pole_x, dtype=nb.float64)
        n = celestial_pole_x.size

    singular_celestial = celestial_pole_x.size == 1
    singular_coordinate = x.size == 1
    singular_native = native_pole_x.size == 1

    for i in range(n):
        coord_i = 0 if singular_coordinate else i
        celes_i = 0 if singular_celestial else i
        nativ_i = 0 if singular_native else i

        theta[i], phi[i] = spherical_project(
            x=x[coord_i],
            y=y[coord_i],
            cos_lat=cos_lat[coord_i],
            sin_lat=sin_lat[coord_i],
            celestial_pole_x=celestial_pole_x[celes_i],
            celestial_pole_y=celestial_pole_y[celes_i],
            celestial_cos_lat=celestial_cos_lat[celes_i],
            celestial_sin_lat=celestial_sin_lat[celes_i],
            native_pole_x=native_pole_x[nativ_i])

    return theta, phi


@nb.njit(cache=True, nogil=False, parallel=False)
def spherical_deproject(phi, theta,
                        celestial_pole_x, celestial_pole_y,
                        celestial_cos_lat, celestial_sin_lat,
                        native_pole_x):  # pragma: no cover
    """
    Deproject single coordinates about a celestial pole to a native pole.

    The following conversions are used depending on the value of the celestial
    pole native latitude (lat_cp) to convert the coordinates (x_cp, y_cp) about
    a celestial pole to coordinates about a native pole (x, y).  Here _cp
    denotes "celestial pole", _np denotes "native pole", and any numbers are in
    degrees.

    lat_cp = 90:

        x = lon_cp + x_cp - lon_np - 180
        y = y_cp

    lat_cp = -90:

        x = lon_cp + lon_np - x_cp
        y = -y_cp

    Otherwise:

        d_cp = x_cp - lon_np
        C = -cos(y_cp)sin(d_cp)
        D = sin(y_cp)cos(lat_cp) - cos(y_cp)sin(lat_cp)cos(d_cp)
        x = lon_cp + arctan2(C, D)
        y = arcsin(sin(y_cp)sin(lat_cp) + cos(y_cp)cos(lat_cp)cos(d_cp))

    The reverse operation (native to celestial poles) can be performed using
    :func:`spherical_project`.

    Parameters
    ----------
    phi : float
        The coordinate longitude in radians about the celestial pole (x_cp).
    theta : float
        The coordinate latitude in radians about the celestial pole (y_cp).
    celestial_pole_x : float
        The celestial pole native longitude in radians (lon_cp).
    celestial_pole_y : float
        The celestial pole native latitude in radians (lat_cp).
    celestial_cos_lat : float
        The cosine of `celestial_pole_y`.
    celestial_sin_lat : float
        The sine of `celestial_pole_y`.
    native_pole_x : float
        The coordinate's native pole longitude in radians.

    Returns
    -------
    x, y : float, float
        The native longitude (x) and latitude (y) coordinates about the native
        pole in radians.
    """

    d_phi = phi - native_pole_x
    right_angle = np.pi / 2

    if equal_angles(np.abs(celestial_pole_y), right_angle):
        if celestial_pole_y > 0:
            cx = celestial_pole_x + d_phi - np.pi
            cy = theta
        else:
            cx = celestial_pole_x - d_phi
            cy = -theta

    else:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_d_phi = np.cos(d_phi)
        cx = celestial_pole_x + np.arctan2(
            -cos_theta * np.sin(d_phi),
            ((sin_theta * celestial_cos_lat)
             - (cos_theta * celestial_sin_lat * cos_d_phi)))
        cy = asin(
            (sin_theta * celestial_sin_lat)
            + (cos_theta * celestial_cos_lat * cos_d_phi))

    return cx, cy


@nb.njit(cache=True, nogil=False, parallel=False)
def spherical_deproject_array(phi, theta,
                              celestial_pole_x, celestial_pole_y,
                              celestial_cos_lat, celestial_sin_lat,
                              native_pole_x):  # pragma: no cover
    """
    Project multiple coordinates from a celestial pole onto a native pole.

    This function is a wrapper around :func:`spherical_deproject` to process
    projections when at least one of either the coordinates, celestial pole, or
    native pole consists of array values.

    Parameters
    ----------
    phi : float or numpy.ndarray
        The longitude coordinates about the celestial pole in radians.  If an
        array is passed in, it should be of shape (1,) or (n,).
    theta : float or numpy.ndarray
        The latitude coordinates about the celestial pole in radians.  If an
        array is passed in, it should be of shape (1,) or (n,).
    celestial_pole_x : float or numpy.ndarray
        The native longitude of the celestial pole.  If an array is passed in,
        it should be of shape (1,) or (n,).
    celestial_pole_y : float or numpy.ndarray
        The native latitude of the celestial pole.  If an array is passed in,
        it should be of shape (1,) or (n,).
    celestial_cos_lat : float or numpy.ndarray
        The cosine of `celestial_pole_y`.
    celestial_sin_lat : float or numpy.ndarray
        The sine of `celestial_pole_y`.
    native_pole_x : float or numpy.ndarray
        The spherical projection's native pole longitude.  If an array
        is passed in, it should be of shape (1,) or (n,).

    Returns
    -------
    x, y : numpy.ndarray, numpy.ndarray
        The projection of (x_cp, y_cp) or (phi, theta) about the celestial pole
        onto the native pole.
    """
    phi = np.atleast_1d(np.asarray(phi))
    theta = np.atleast_1d(np.asarray(theta))
    celestial_pole_x = np.atleast_1d(np.asarray(celestial_pole_x))
    celestial_pole_y = np.atleast_1d(np.asarray(celestial_pole_y))
    celestial_cos_lat = np.atleast_1d(np.asarray(celestial_cos_lat))
    celestial_sin_lat = np.atleast_1d(np.asarray(celestial_sin_lat))
    native_pole_x = np.atleast_1d(np.asarray(native_pole_x))

    sizes = np.array([phi.size, celestial_pole_x.size, native_pole_x.size])
    max_array = np.argmax(sizes)
    if max_array == 0:
        x = np.empty_like(phi, dtype=nb.float64)
        y = np.empty_like(phi, dtype=nb.float64)
        n = phi.size
    else:
        x = np.empty_like(celestial_pole_x, dtype=nb.float64)
        y = np.empty_like(celestial_pole_x, dtype=nb.float64)
        n = celestial_pole_x.size

    flat_x, flat_y = x.flat, y.flat
    flat_phi, flat_theta = phi.flat, theta.flat
    flat_cx, flat_cy = celestial_pole_x.flat, celestial_pole_y.flat
    flat_celestial_cos_lat = celestial_cos_lat.flat
    flat_celestial_sin_lat = celestial_sin_lat.flat

    singular_celestial = celestial_pole_x.size == 1
    singular_theta = theta.size == 1
    singular_native = native_pole_x.size == 1

    for i in range(n):
        theta_i = 0 if singular_theta else i
        celes_i = 0 if singular_celestial else i
        nativ_i = 0 if singular_native else i
        flat_x[i], flat_y[i] = spherical_deproject(
            phi=flat_phi[theta_i],
            theta=flat_theta[theta_i],
            celestial_pole_x=flat_cx[celes_i],
            celestial_pole_y=flat_cy[celes_i],
            celestial_cos_lat=flat_celestial_cos_lat[celes_i],
            celestial_sin_lat=flat_celestial_sin_lat[celes_i],
            native_pole_x=native_pole_x[nativ_i])

    return x, y


@nb.njit(cache=True, nogil=False, parallel=False)
def calculate_celestial_pole(native_reference_x, native_reference_cos_lat,
                             native_reference_sin_lat,
                             reference_x, reference_y,
                             reference_cos_lat, reference_sin_lat,
                             native_pole_x, native_pole_y,
                             select_solution):  # pragma: no cover
    """
    Calculate a celestial pole based on a reference and native reference.

    The determination of a celestial pole may involve a few steps.  The first
    is to determine the reference position about the native pole wrt the native
    reference.  The next step involves finding the positions of the northern
    and southern poles.

    A number of solutions are available at this point, and may be selected via
    the `select_solution` parameter value:

        -1 : Always select the southern pole.
         1 : Always select the northern pole.
         0 : Select the pole closest to the native pole.

    If only one pole is found, it will always be chosen regardless of the
    selection method.  If no pole is found, NaN values for the celestial pole
    coordinates will be returned.

    Parameters
    ----------
    native_reference_x : float
        The longitude of the native reference position in radians.
    native_reference_cos_lat : float
        The cosine of the native reference latitude.
    native_reference_sin_lat : float
        The sine of the native reference latitude.
    reference_x : float
        The reference position longitude in radians.
    reference_y : float
        The reference position latitude in radians.
    reference_cos_lat : float
        The cosine of the reference latitude.
    reference_sin_lat : float
        The sine of the reference latitude.
    native_pole_x : float
        The native pole longitude in radians.
    native_pole_y : float
        The native pole latitude in radians.
    select_solution : int
        The method by which to choose the orientation of the celestial pole.
        -1 = "southern", 1 = "northern", 0 = "nearest".

    Returns
    -------
    celestial_pole_longitude, celestial_pole_latitude : float, float
        The celestial pole longitude and latitude in radians.
    """
    right_angle = np.pi / 2
    d_phi = native_pole_x - native_reference_x
    sin_d_phi = np.sin(d_phi)
    cos_d_phi = np.cos(d_phi)
    delta_p1 = np.arctan2(
        native_reference_sin_lat, native_reference_cos_lat * cos_d_phi)
    cs = native_reference_cos_lat * sin_d_phi

    delta_p2 = acos(reference_sin_lat / np.sqrt(1 - (cs ** 2)))
    celestial_y = 0.0

    delta_n = delta_p1 + delta_p2
    delta_s = delta_p1 - delta_p2
    if delta_n > delta_s:
        temp = delta_s
        delta_s = delta_n
        delta_n = temp

    solutions = 0
    if np.abs(delta_n) <= right_angle:
        celestial_y = delta_n
        solutions += 1

    if np.abs(delta_s) <= right_angle:
        solutions += 1
        if solutions == 1:
            celestial_y = delta_s
        elif select_solution == -1:
            celestial_y = delta_s
        elif select_solution == 0:
            if np.abs(delta_s - native_pole_y) < np.abs(
                    delta_n - native_pole_y):
                celestial_y = delta_s

    if solutions == 0:  # pragma: no cover (shouldn't happen)
        return np.nan, np.nan

    if equal_angles(np.abs(reference_y), right_angle):
        celestial_x = reference_x
    elif equal_angles(np.abs(celestial_y), right_angle):
        celestial_x = reference_x
        if celestial_y > 0:
            celestial_x += native_pole_x - native_reference_x - np.pi
        else:
            celestial_x += native_reference_x - native_pole_x
    else:
        cl = np.cos(celestial_y)
        sl = np.sin(celestial_y)

        sin_d_lon = sin_d_phi * native_reference_cos_lat / reference_cos_lat
        cos_d_lon = native_reference_sin_lat - (sl * reference_sin_lat)
        cos_d_lon /= cl * reference_cos_lat
        celestial_x = reference_x - np.arctan2(sin_d_lon, cos_d_lon)

    return celestial_x, celestial_y


@nb.njit(cache=True, nogil=False, parallel=False)
def calculate_celestial_pole_array(native_reference_x,
                                   native_reference_cos_lat,
                                   native_reference_sin_lat,
                                   reference_x, reference_y,
                                   reference_cos_lat, reference_sin_lat,
                                   native_pole_x, native_pole_y,
                                   select_solution):  # pragma: no cover
    """
    Calculate the celestial pole when one or more of the inputs are arrays.

    This is a wrapper around :func:`calculate_celestial_pole` for use when
    processing array values.

    Parameters
    ----------
    native_reference_x : float or numpy.ndarray
        The native reference longitude in radians.  If an array is provided, it
        should be of shape (n,).
    native_reference_cos_lat : float or numpy.ndarray
        The cosine of the native reference latitude.  Must be the same input
        shape as `native_reference_x`.
    native_reference_sin_lat : float or numpy.ndarray
        The sine of the native reference latitude.  Must be the same input
        shape as `native_reference_x`.
    reference_x : float or numpy.ndarray
        The reference longitude in radians.  If an array is provided, it should
        be of shape (n,).
    reference_y : float or numpy.ndarray
        The reference latitude in radians.  If an array is provided, it should
        be of shape (n,).
    reference_cos_lat : float or numpy.ndarray
        The cosine of the reference latitude.  Must be the same input shape as
        `reference_y`.
    reference_sin_lat : float or numpy.ndarray
        The sine of the reference latitude.  Must be the same input shape as
        `reference_y`.
    native_pole_x : float or numpy.ndarray
        The native pole longitude in radians.  If an array is provided, it
        should be of shape (n,).
    native_pole_y : float or numpy.ndarray
        The native pole latitude in radians.  If an array is provided, it
        should be of shape (n,).
    select_solution : int
        The celestial pole to choose.  1 for "northern", 2 for "southern", or
        0 for "nearest".

    Returns
    -------
    celestial_pole_longitude, celestial_pole_latitude : np.ndarray, np.ndarray
        The celestial pole longitude and latitude in radians of shape (1,) or
        (n,) depending on whether any array-like values were passed in.
    """
    native_reference_x = np.atleast_1d(np.asarray(native_reference_x))
    native_reference_cos_lat = np.atleast_1d(
        np.asarray(native_reference_cos_lat))
    native_reference_sin_lat = np.atleast_1d(
        np.asarray(native_reference_sin_lat))
    reference_x = np.atleast_1d(np.asarray(reference_x))
    reference_y = np.atleast_1d(np.asarray(reference_y))
    reference_cos_lat = np.atleast_1d(np.asarray(reference_cos_lat))
    reference_sin_lat = np.atleast_1d(np.asarray(reference_sin_lat))
    native_pole_x = np.atleast_1d(np.asarray(native_pole_x))
    native_pole_y = np.atleast_1d(np.asarray(native_pole_y))

    sizes = np.array([native_reference_x.size,
                      reference_x.size,
                      native_pole_x.size])
    max_array = np.argmax(sizes)

    if max_array == 0:
        x = np.empty_like(native_reference_x, dtype=nb.float64)
        y = np.empty_like(native_reference_x, dtype=nb.float64)
    elif max_array == 1:
        x = np.empty_like(reference_x, dtype=nb.float64)
        y = np.empty_like(reference_x, dtype=nb.float64)
    else:
        x = np.empty_like(native_pole_x, dtype=nb.float64)
        y = np.empty_like(native_pole_x, dtype=nb.float64)

    n = x.size

    flat_x, flat_y = x.flat, y.flat
    flat_native_reference_x = native_reference_x.flat
    flat_native_reference_cos_lat = native_reference_cos_lat.flat
    flat_native_reference_sin_lat = native_reference_sin_lat.flat
    flat_reference_x = reference_x.flat
    flat_reference_y = reference_y.flat
    flat_reference_cos_lat = reference_cos_lat.flat
    flat_reference_sin_lat = reference_sin_lat.flat
    flat_native_pole_x = native_pole_x.flat
    flat_native_pole_y = native_pole_y.flat

    singular_native_reference = native_reference_x.size == 1
    singular_reference = reference_x.size == 1
    singular_native_pole = native_pole_x.size == 1

    for i in range(n):
        natref_i = 0 if singular_native_reference else i
        ref_i = 0 if singular_reference else i
        pole_i = 0 if singular_native_pole else i

        flat_x[i], flat_y[i] = calculate_celestial_pole(
            native_reference_x=flat_native_reference_x[natref_i],
            native_reference_cos_lat=flat_native_reference_cos_lat[natref_i],
            native_reference_sin_lat=flat_native_reference_sin_lat[natref_i],
            reference_x=flat_reference_x[ref_i],
            reference_y=flat_reference_y[ref_i],
            reference_cos_lat=flat_reference_cos_lat[ref_i],
            reference_sin_lat=flat_reference_sin_lat[ref_i],
            native_pole_x=flat_native_pole_x[pole_i],
            native_pole_y=flat_native_pole_y[pole_i],
            select_solution=select_solution)
    return x, y
