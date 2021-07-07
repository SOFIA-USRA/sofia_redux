# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate response ratio for atmospheric correction."""

from numpy import cos, log, pi, poly1d

__all__ = ['pipecal_rratio']


def pipecal_rratio(za, altwv, za_ref, altwv_ref,
                   za_coeff, altwv_coeff, pwv=False):
    """
    Calculate the R ratio for a given ZA and Altitude or PWV.

    The R ratio is used to correct image flux values from the observed
    atmospheric conditions to a reference atmospheric condition.  This
    is required before applying calibration factors calculated at the
    reference atmosphere.

    The procedure is:
        1. Calculate the ZA term from the polynomial coefficients
           and sec(ZA) - sec(ZAref).
        2. Calculate the altitude or PWV term from the polynomial
           coefficients and log(PWV) - log(PWVref),
           or ALT - ALTref.
        3. Multiply the terms together and return the result.

    Parameters
    ----------
    za : float
        Observed zenith angle (degrees).
    altwv : float
        Observed altitude (kilo-feet) or precipitable water vapor (um).
    za_ref : float
        Reference ZA used to calculate response fits (degrees).
    altwv_ref : float
        Reference altitude (kilo-feet) or PWV (um) used to
        calculate response fits.
    za_coeff : list of float
        Polynomial coefficients of response fit in ZA.
    altwv_coeff : list of float
        Polynomial coefficients of response fit in altitude or PWV.
    pwv : bool, optional
        If set, the water vapor response fit is used instead
        of the altitude fit, and altwv is treated as a water
        vapor value.

    Returns
    -------
    rratio : float
        R ratio at observed ZA and altitude/PWV.
    """

    # Calculate the polynomial for the ZA term
    seczref = 1.0 / cos(pi / 180. * za_ref)
    secz = 1.0 / cos(pi / 180. * za)
    dsecz = secz - seczref

    # Make a polynomial with the coefficients of
    # za_coeff. The array is reversed since:
    #    IDL: poly(x, [1,2,3]) = 1 + 2x + 3x**2
    #    numpy: poly1d([1,2,3]) = 1x**2 + 2x + 3
    # The coefficients are listed with the constant term first.
    p = poly1d(za_coeff[::-1])
    seczterm = p(dsecz)

    # Calculate the altitude or WV term
    p = poly1d(altwv_coeff[::-1])
    if pwv:
        altwvterm = p(log(altwv / altwv_ref))
    else:
        altwvterm = p(altwv - altwv_ref)

    # Multiply the terms
    rratio = seczterm * altwvterm
    return rratio
