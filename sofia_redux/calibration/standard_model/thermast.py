# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Generate a themeral model of an asteroid"""

import numpy as np
from scipy import integrate as si


def thermast(Gmag, pV, rsize, phang, dsun, dist,
             nw=500):
    """
    Create a model of the thermal flux from an asteroid.

    Follow the process outlined in Delbo and Harris, 2002.

    Parameters
    ----------
    Gmag : float
        Magnitude slope constant.
    pV : float
        Albedo of the asteroid in V band.
    rsize : float
        Radius of the asteroid in km.
    phang : float
        Phase angle of the asteroid in degrees.
    dsun : float
        Distance between the Sun and the asteroid in AU.
    dist : float
        Distance between the Earth and the asteroid in AU.
    nw : int, optional
        Number of wavelength data points to generate.
        Defaults to 2500.

    Returns
    -------
    warr : numpy.array
        Array of length `nw` containing the wavelength data.
    flux : numpy.arry
        Array of length `nw` containing the flux emitted by
        the asteroid at each wavelength in `warr`.

    """
    # Define constants
    c = 2.9979e14  # micros/sec
    stefan_boltzmann = 5.67051e-5  # erg/cm2/s/K^4
    solar_constant = 0.1368e7  # erg/cm2/s
    au = 1.49598e8  # km / 1 AU
    W2Jy = 1.0e-26  # W/m2/Hz/Jy

    eta = 0.756  # beaming factor
    emissivity = 0.90  # emissivity

    correction = correction_factor(phase_angle=phang)

    # Convert distance from AU to km
    dist *= au

    # Create wavelength array
    wmin = 1.0
    wmax = 300.0
    warr = np.linspace(wmin, wmax, nw)

    qint = 0.290 + 0.684 * Gmag
    alb_bond = qint * pV
    subsolar_temp_max = ((1.0 - alb_bond) * solar_constant
                         / (emissivity * stefan_boltzmann * eta
                            * dsun ** 2)) ** 0.25
    omin = 0.
    omax = np.pi / 2.

    # For each wavelength, integrate over angle to get the
    # total flux
    fnu = flux_at_w(warr, subsolar_temp_max, omin, omax)
    fnu *= 2. * np.pi * emissivity * correction * (rsize / dist) ** 2
    fnu *= warr * warr / (c * W2Jy)

    return warr, fnu


def correction_factor(phase_angle):
    """
    Calculate correction factor from asteroid's phase angle.

    Allows the STM to be used at non-zero solar phase angles,
    taken from Lebofsky et al. (1986).

    Parameters
    ----------
    phase_angle : float
        Phase angle of the asteroid in degrees.

    Returns
    -------
    correction : float
        Correction factor

    """
    correction = 10.0 ** (-0.01 * np.abs(phase_angle) / 2.5)
    return correction


def flux_at_w(warr, subsolar_temperature, omin=0., omax=np.pi / 2.):
    r"""
    Calculate the thermal flux from a blackbody object.

    Parameters
    ----------
    warr : numpy.array
        Wavelengths to calculate flux at, in microns.
    subsolar_temperature : float
        Maximum sub-solar temperature of the asteroid,
        as given by Equation 9 in Deblo & Harris (2002).
    omin : float, optional
        Lower omega limit of integration in radians.
        Defaults to 0.
    omax : float
        Upper omega limit of integration in radians.
        Defaults to $\pi$/2.

    Returns
    -------
    flux : numpy.array
        Wavelength dependent component of the blackbody
        flux of the object.

    Notes
    -----
    Implementation of the integral in Equation 10 from
    Delbo & Harris (2002).

    .. math::

        F_{\lambda} \,=\, \int_{\Omega_{min}}^{\Omega{\max}} d\Omega


    """
    if omin < 0:
        omin = 0
    if omax > np.pi / 2:
        omax = np.pi / 2
    flux = np.zeros_like(warr)
    for i, w in enumerate(warr):
        flux[i], _ = si.fixed_quad(bbflux, omin, omax, n=10,
                                   args=(subsolar_temperature, w))
    return flux


def bbflux(omega, tss, w):
    """
    Calculate the modified blackbody flux from the asteroid.

    Parameters
    ----------
    omega : numpy.array
        Angular distance from the sub-solar point.
    tss : float
        Maximum sub-solar temperature of the asteroid in Kelvins.
    w : float
        Wavelength in microns.

    Returns
    -------
    flux :  numpy.array
        Thermal flux as a function of `omega`.

    Notes
    -----
    This function computes the integrand of Equation 10
    from Delbo & Harris (2002).

    """
    temp = tss * np.cos(omega) ** 0.25
    flux = planck_function(w, temp) * np.cos(omega) * np.sin(omega)
    return flux


def planck_function(w, temp):
    """
    Evaluate the Planck function.

    Parameters
    ----------
    w : float, numpy.array
        Wavelength in microns.
    temp : float, numpy.array
        Temperature of object in Kelvins.

    Returns
    -------
    bb : float, numpy.array
        The blackbody flux from the object in
        W/m2/micron/str.

    """
    # Constants
    h = 6.6261e-27  # ergs-s
    c = 2.9979e10  # cm/s
    k = 1.3897e-16  # erg/K

    # Conversions
    mu2cm = 1.0e-4  # cm/micron
    erg2w = 1.0e-7  # W/(erg/s)
    m2cm = 1.0e2  # cm/m

    # Convert wavelength to cm
    wcm = w * mu2cm
    term1 = 2.0 * h * c * c / (w * (wcm ** 4))
    term1 = term1 * erg2w * m2cm * m2cm
    zterm = h * c / (wcm * k * temp)

    idx = zterm >= 150.

    term2 = np.exp(zterm) - 1.
    bb = term1 / term2
    if isinstance(bb, np.ndarray):
        bb[bb < 0] = 0.
        bb[idx] = 0.

    return bb
