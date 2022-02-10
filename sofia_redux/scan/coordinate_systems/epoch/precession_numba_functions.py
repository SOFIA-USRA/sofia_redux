# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numba as nb
import numpy as np

nb.config.THREADING_LAYER = 'threadsafe'

__all__ = ['precess_single', 'precess_times']


@nb.njit(cache=True, nogil=False, parallel=False)
def precess_single(p, ra, dec, cos_lat, sin_lat):  # pragma: no cover
    """
    Precess the coordinates with a precession matrix.

    RA and DEC coordinates are updated in-place according to::

       v0 = cos(DEC) * cos(RA)
       v1 = cos(DEC) * sin(RA)
       v2 = sin(DEC)

       l = p X |v0 v1 v2| (where X indicates matrix multiplication)

       RA_precessed = arctan2(l[1], l[0])
       DEC_precessed = arctan2(l[2], sqrt(l[0]^2 + l[1]^2))

    Parameters
    ----------
    p : numpy.ndarray (float)
        The precession matrix of shape (3, 3)
    ra : numpy.ndarray (float)
        The Right Ascension coordinates in radians of shape (shape).
    dec : numpy.ndarray (float)
        The Declination coordinates in radians of shape (shape).
    cos_lat : numpy.ndarray (float)
        The cosine(latitude) values of the coordinates of shape (shape).
    sin_lat : numpy.ndarray (float)
        The sine(latitude) values of the coordinates of shape (shape).

    Returns
    -------
    None
    """
    n = ra.size
    x = np.empty(3, dtype=nb.float64)
    flat_ra = ra.flat
    flat_dec = dec.flat
    flat_cos_lat = cos_lat.flat
    flat_sin_lat = sin_lat.flat
    for i in range(n):
        ra_i = flat_ra[i]
        cos_lat_i = flat_cos_lat[i]
        v0 = cos_lat_i * np.cos(ra_i)
        v1 = cos_lat_i * np.sin(ra_i)
        v2 = flat_sin_lat[i]
        for row in range(3):
            x[row] = (p[row, 0] * v0) + (p[row, 1] * v1) + (p[row, 2] * v2)
        flat_ra[i] = np.arctan2(x[1], x[0])
        flat_dec[i] = np.arctan2(x[2], np.hypot(x[0], x[1]))


@nb.njit(cache=True, nogil=False, parallel=False)
def precess_times(p, ra, dec, cos_lat, sin_lat):  # pragma: no cover
    """
    Precess the coordinates with a precession matrix.

    RA and DEC coordinates are updated in-place.  Please see
    :func:`precess_single` for further details on the algorithm.

    Parameters
    ----------
    p : numpy.ndarray (float)
        The precession matrix of shape (shape, 3, 3)
    ra : numpy.ndarray (float)
        The Right Ascension coordinates in radians of shape (shape).
    dec : numpy.ndarray (float)
        The Declination coordinates in radians of shape (shape).
    cos_lat : numpy.ndarray (float)
        The cosine(latitude) values of the coordinates of shape (shape).
    sin_lat : numpy.ndarray (float)
        The sine(latitude) values of the coordinates of shape (shape).

    Returns
    -------
    None
    """
    n = ra.size
    x = np.empty(3, dtype=nb.float64)
    flat_ra = ra.flat
    flat_dec = dec.flat
    flat_cos_lat = cos_lat.flat
    flat_sin_lat = sin_lat.flat
    flat_p = p.reshape((n, 3, 3))
    for i in range(n):
        p_i = flat_p[i]
        ra_i = flat_ra[i]
        cos_lat_i = flat_cos_lat[i]
        v0 = cos_lat_i * np.cos(ra_i)
        v1 = cos_lat_i * np.sin(ra_i)
        v2 = flat_sin_lat[i]
        for row in range(3):
            x[row] = ((p_i[row, 0] * v0)
                      + (p_i[row, 1] * v1)
                      + (p_i[row, 2] * v2))
        flat_ra[i] = np.arctan2(x[1], x[0])
        flat_dec[i] = np.arctan2(x[2], np.hypot(x[0], x[1]))
