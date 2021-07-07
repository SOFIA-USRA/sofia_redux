# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import warnings

__all__ = ['nlambda']


def nlambda(wavelength, pressure, temperature, water=0.0):
    """
    Compute the real part of the refractive index of air.

    Based on the formulas in Filippenko's article in
    1982, PASP, 94, 715.

    Parameters
    ----------
    wavelength : float or numpy.ndarray of float (N,)
        Wavelength of light in microns
    pressure : float or numpy.ndarray of float (N,)
        Atmospheric pressure in mm of Hg
    temperature : float or numpy.ndarray of float (N,)
        Atmospheric temperature in degrees Celsius
    water : float of numpy.ndarray (N,), optional
        Water vapour pressure in mm of Hg

    Returns
    -------
    index_of_refraction : float or numpy.ndarray of float (N,)
        The index of refraction for the input conditions
    """

    with warnings.catch_warnings():
        wavenumber = 1 / wavelength
        wn2 = wavenumber ** 2
        stp = 64.328 + 29498.1 / (146 - wn2) + 255.4 / (41 - wn2)
        pt_correction = 1 + (1.049 - (0.0157 * temperature)) * 1e-6 * pressure
        pt_correction *= pressure
        pt_correction /= (720.883 * (1.0 + (3.661e-3 * temperature)))
        if not np.allclose(water, 0):
            water *= (0.0624 - (6.8e-4 * wn2)) / (1 + 3.661e-3 * temperature)

    index = 1 + (stp * pt_correction - water) / 1e6
    return index
