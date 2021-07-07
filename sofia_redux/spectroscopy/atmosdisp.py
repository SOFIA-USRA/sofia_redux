# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from sofia_redux.spectroscopy.nlambda import nlambda
import warnings

__all__ = ['atmosdisp']


def atmosdisp(wavelength, refwave, za, pressure, temperature,
              water=0.0, altitude=0):
    """
    Compute the atmospheric dispersion relative to wave0

    Computes the difference between the dispersion at two
    wavelengths.  The dispersion for each wavelength is derived
    from Section 4.3 of Green's "Spherical Astronomy" (1985).

    Parameters
    ----------
    wavelength : float or numpy.ndarray of float (N,)
        Wavelength in microns
    refwave : float or numpy.ndarray of float (N,)
        Reference wavelength in microns
    za : float or numpy.ndarray of float (N,)
        Zenith angle of object
    pressure : float or numpy.ndarray of float (N,)
        Atmospheric pressure in mm of Hg
    temperature : float or numpy.ndarray of float (N,)
        Atmospheric temperature in degrees C
    water : float or numpy.ndarray of float (N,), optional
        Water vapour pressure in mm of Hg
    altitude : float or numpy.ndarray of float (N,), optional
        Observatory altitude in km

    Returns
    -------
    dispersion : float or numpy.ndarray of float (N,)
        The atmospheric dispersion in arcseconds
    """
    mean_earth_radius = 6371.03  # km
    hconst = 0.02926554  # R/(mu*g) in km/deg K, R = gas const=8.3143e7
    tempk = temperature + 273.15  # K

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        hratio = (hconst * tempk) / (mean_earth_radius + altitude)

        # Compute index of refraction
        ind = nlambda(wavelength, pressure, temperature, water)
        rind = nlambda(refwave, pressure, temperature, water)

        # Compute dispersion
        a = (1.0 - hratio) * (ind - rind)
        b = 0.5 * (ind ** 2 - rind ** 2) - (1 + hratio) * (ind - rind)
        tan_za = np.tan(np.radians(za))
        dispersion = 206265 * tan_za * (a + b * tan_za ** 2)

    return dispersion
