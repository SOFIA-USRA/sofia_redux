# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate various isophotal wavelengths"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline


def calc_isophotal(result, wavelengths, pl, alpha, model_flux_in_filter,
                   atmosphere_transmission_in_filter, total_throughput):
    """
    Calculated the isophotal wavelengths.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    wavelengths : numpy.array
        Wavelengths in the band being examined.
    pl : bool
        If True, the model is a power law.
    alpha : float
        The index used for the power law model if `pl` is True.
    model_flux_in_filter : numpy.array
        The modeled flux in the current filter.
    atmosphere_transmission_in_filter : numpy.array
        The atmospheric transmission in the current filter.
    total_throughput : float
        The throughput through the telescope, instrument, and optics.

    Returns
    -------
    result : dict
        Same as the input `result` but with 'isophotal' and 'isophotal_wt'
        keys populated.

    """
    # Find isophotal wavelengths
    imin = np.argmin(wavelengths)
    imax = np.argmax(wavelengths)

    # fzero = model_flux_in_filter[imin:imax + 1] - result['flux_mean']
    warrz = wavelengths[imin:imax + 1]
    flux = model_flux_in_filter[imin:imax + 1]

    if pl:
        iso_waves = isophotal_from_powerlaw(result, alpha, wavelengths, flux,
                                            total_throughput,
                                            atmosphere_transmission_in_filter,
                                            model_flux_in_filter)
    else:
        fzero = flux - result['flux_mean']
        wave_zeros = find_isophotal_candidates(warrz, fzero)
        iso_waves = isophotal_from_zeros(wave_zeros,
                                         atmosphere_transmission_in_filter,
                                         total_throughput,
                                         wavelengths, model_flux_in_filter)

    result['isophotal'], result['isophotal_wt'] = iso_waves

    return result


def find_isophotal_candidates(warr, flux_reduced):
    """
    Find where the flux is equal to the mean flux.

    Parameters
    ----------
    warr : numpy.array
        Wavelengths in the filter.
    flux : numpy.array
        The flux at each wavelength in `warr`.

    Returns
    -------
    zeros : numpy.array
        Wavelengths where flux = mean(flux)

    """

    # mean_flux = np.nanmean(flux)
    # flux_reduced = flux - mean_flux
    f = UnivariateSpline(warr, flux_reduced, s=0)
    zeros = f.roots()
    return zeros


def isophotal_from_powerlaw(result, alpha, warr=None, flux=None,
                            total_throughput=None, taf=None, fsi=None):
    """
    Calculate the isophotal wavelengths for a power law model.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    alpha : float
        Slope to use for power-law model.
    warr : numpy.array
        Wavelengths in the current filter.
    flux : numpy.array
        Flux at each wavelength in `warr`.
    total_throughput : float
        Total throughput through telescope + instrument + optics.
    taf : numpy.array
        Transmission in the atmosphere in the current filter.
    fsi : numpy.array
        Modeled flux in the current filter.

    Returns
    -------
    isophotal : float
        Isophotal wavelength.
    isophotal_weight : float
        Weighted isophotal wavelength.

    """
    if alpha == -2:
        isophotal = result['lambda_mean']
        isophotal_weight = result['lambda_mean']
    else:
        wave_zeros = find_isophotal_candidates(warr, flux)
        if len(wave_zeros) > 1:
            isophotal, isophotal_weight = \
                isophotal_from_zeros(wave_zeros, taf, total_throughput,
                                     warr, fsi)
        else:
            isophotal = wave_zeros
            isophotal_weight = wave_zeros
    return isophotal, isophotal_weight


def isophotal_from_zeros(wave_zeros, taf, total_throughput, warr, fsi):
    """
    Calculate the isophotal wavelengths from zero points.

    Parameters
    ----------
    wave_zeros : float, list, numpy.array
        Wavelengths where the flux is equal to the mean flux
        in the filter.
    taf : numpy.array
        Atmospheric transmission in the filter.
    total_throughput : float
        Total throughput through telescope + instrument + optics.
    warr : numpy.array
        Wavelengths in the current filter.
    fsi : numpy.array
        Modeled flux in the current filter.

    Returns
    -------
    isophotal : float
        Isophotal wavelength.
    isophotal_weight : float
        Weighted isophotal wavelength.

    """
    if isinstance(wave_zeros, (list, np.ndarray)):
        if len(wave_zeros) > 1:
            isophotal = np.mean(wave_zeros)
            tfzero = interpol(fsi * total_throughput * taf, warr, wave_zeros)
            lam_iso_wt2 = np.sum(wave_zeros * tfzero) / np.sum(tfzero)
            isophotal_weight = lam_iso_wt2
        else:
            isophotal = wave_zeros[0]
            isophotal_weight = wave_zeros[0]
    else:
        isophotal = wave_zeros
        isophotal_weight = wave_zeros
    return isophotal, isophotal_weight


def interpol(v, x, xout):
    """Interpolates v(x) at points xout."""
    fit = interp1d(x, v, fill_value='extrapolate')
    return fit(xout)


def calculated_lambdas(result, integrals):
    """
    Calculate various characteristic wavelengths.

    Parameters
    ----------
    result : dict, pandas.Series
        Collection of various calibration results.
    integrals : dict
        Collection of various background integrals.

    Returns
    -------
    result : dict
        Same as input `result` but with several wavelength
        keys populated.

    References
    ----------
    Wavelengths are defined in Tokunaga & Vacca 2005.

    """

    result['lambda_1'] = integrals['10'] / integrals['2']
    result['nuref'] = integrals['3'] / integrals['11']
    result['irac'] = integrals['1'] / integrals['3']
    result['sdss'] = np.exp(integrals['7'] / integrals['3'])
    result['eff_ph'] = integrals['6'] / integrals['5']

    result['lambda_mean'] = integrals['2'] / integrals['1']
    result['lambda_eff'] = integrals['5'] / integrals['4']
    result['lambda_pivot'] = np.sqrt(integrals['1'] / integrals['11'])
    result['lambda_eff_jv'] = integrals['11'] / integrals['12']
    result['rms'] = np.sqrt(integrals['10'] / integrals['3']
                            - result['lambda_mean'] ** 2)
    result['lambda_prime'] = integrals['1'] / integrals['3']

    result['lamcorr'] = (result['lambda_prime'] * result['lambda_mean']
                         / result['lambda_pivot']**2)

    return result
