# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from sofia_redux.toolkit.fitting.polynomial import poly1d
import numpy as np

__all__ = ['si_index_of_refraction']


def si_index_of_refraction(wave, temperature):
    """
    Return the index of refraction for Si

    Generates the index of refraction using the formula's in
    Frey et al. (2006, SPIE, 6273, 2J).

    Parameters
    ----------
    wave : float or array_like of float (shape)
        Wavelength in microns
    temperature : float or array_like of float (shape)
        The temperature in Kelvin

    Returns
    -------
    siior : numpy.ndarray of float (shape)
        The index of refraction of Si at the requested wavelengths
    """
    t_arr = hasattr(temperature, '__len__')
    w_arr = hasattr(wave, '__len__')
    is_arr = t_arr or w_arr
    temperature = np.asarray(temperature, dtype=float)
    wave = np.asarray(wave, dtype=float)
    if t_arr and w_arr and wave.shape != temperature.shape:
        raise ValueError("wave and temperature shape mismatch")
    invalid = (wave < 1.1) | (wave > 5.6)
    if invalid.any():
        log.warning("Values only good for 1.1 < wavelength < 5.6 um")
    invalid = np.asarray((temperature < 20) | (temperature > 300))
    if invalid.any():
        log.warning("Values only good for 20 < temperature < 300 K")

    sc = np.array([
        [10.4907, -2.08020e-4, 4.21694e-6, -5.82298e-9, 3.44688e-12],
        [-1346.61, 29.1664, -0.278724, 1.05939e-3, -1.35089e-6],
        [4.42827e7, -1.76213e6, -7.61575e4, 678.414, 103.243]
    ])
    lc = np.array([
        [0.299713, -1.14234e-5, 1.67134e-7, -2.51049e-10, 2.32484e-14],
        [-3.51710e3, 42.3892, -0.357957, 1.17504e-3, -1.13212e-6],
        [1.71400e6, -1.44984e5, -6.9074e3, -39.3699, 23.5770]
    ])
    s_stack = np.vstack((poly1d(temperature[None], sc[0]),
                         poly1d(temperature[None], sc[1]),
                         poly1d(temperature[None], sc[2])))
    l_stack = np.vstack((poly1d(temperature[None], lc[0]),
                         poly1d(temperature[None], lc[1]),
                         poly1d(temperature[None], lc[2])))
    wave2 = wave ** 2
    l2 = l_stack ** 2
    n21 = s_stack[0] * wave2 / (wave2 - l2[0])
    n21 += s_stack[1] * wave2 / (wave2 - l2[1])
    n21 += s_stack[2] * wave2 / (wave2 - l2[2])

    result = np.sqrt(n21 + 1)
    if not is_arr:
        result = result[0]

    return result
