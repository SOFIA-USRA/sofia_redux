# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Calculate properties of observations derived from the models"""

import os
import numpy as np
import scipy.integrate as si
import astropy.constants as const
import astropy.units as u

from sofia_redux.calibration.standard_model import isophotal_wavelength as iso
from sofia_redux.calibration.standard_model import thermast


def mean_fluxes(result, integrals):
    """
    Calculate the mean flux.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    integrals : dict
        Various integrals.

    Returns
    -------
    result : dict
        Same as input `result` but with the mean flux in F_lambda
        and F_nu populated.

    """
    c = const.c.to(u.um / u.s).value
    Jy2W = 1e-26  # Convert Jy to W/m2/Hz
    f_mean = integrals['4'] / integrals['1']
    fnu_mean = f_mean * result['lambda_pivot'] ** 2 / (c * Jy2W)
    result['flux_mean'], result['flux_nu_mean'] = f_mean, fnu_mean
    return result


def mean_pixels_in_beam(num_pix, total_throughput, wavelengths):
    """
    Calculate the mean number of pixels in the beam.

    Mean is weighted by the total_throughput at each wavelength.

    Parameters
    ----------
    num_pix : ndarray
        Number of pixels in extraction area at each wavelength.
    total_throughput : ndarray
        Total throughput of the telescope at each wavelength.
    wavelengths : ndarray
        Wavelenths to integrate over in microns.

    Returns
    -------
    npix_mean : float
        Throughput-weighted mean number of pixels in
        the beam.

    """
    top = si.simps(num_pix * total_throughput, wavelengths)
    bottom = si.simps(total_throughput, wavelengths)
    npix_mean = top / bottom
    return npix_mean


def noise_equivalent_power(bg_integrand1, bg_integrand2, num_pix,
                           wavelengths, omega_pix, telescope_area):
    """
    Calculate the noise-equivalent power.

    This is the power of the source required to general a signal
    equivalent to the noise.

    Parameters
    ----------
    bg_integrand1 : float
        First background integrand.
    bg_integrand2 : numpy.ndarray
        Second background integrand.
    num_pix : numpy.ndarray
        Number of pixels in the beam as a function of wavelengths.
    wavelengths : numpy.ndarray
        Wavelenghts to integrate over in microns.
    omega_pix : float
        Pixel solid angle in steradians
    telescope_area : float
        Area of the telescope in microns.

    Returns
    -------
    nep : float
        Noise equivalent power in Watts.

    """
    ergs2W = 1e-7  # Convert ergs/s to Watts
    c = const.c.to(u.um / u.s).value
    h = const.h.to(u.erg * u.s).value

    nepterm1 = si.simps(bg_integrand1 * num_pix / wavelengths, wavelengths)
    nepterm2 = si.simps(bg_integrand2 * num_pix / wavelengths, wavelengths)
    nep = np.sqrt(2. * telescope_area * omega_pix * h * ergs2W * c
                  * (nepterm1 + nepterm2))

    return nep


def limiting_flux(result, integrals, snr_ref, tref=900.):
    """
    Calculate the minimum observable flux.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    integrals : dict
         Collection of various background integrals.
    snr_ref : float
        Signal-to-noise.
    tref : float, optional
        Reference transmission. Defaults to 900.

    Returns
    -------
    result : dict
        Same as input `result`, but with NEFD (noise-equivalent
        flux density) and MDCF (limiting flux) populated.

    """
    # Compute SNR and limiting flux
    c = const.c.to(u.um / u.s).value
    Jy2W = 1e-26  # Convert Jy to W/m2/Hz

    nefd = result['nep'] / (c * integrals['11'])
    nefd /= Jy2W

    snr = result['flux_nu_mean'] / (nefd / np.sqrt(2.))
    flim = snr_ref * result['flux_nu_mean'] * 1000. / (snr * np.sqrt(tref))

    result['nefd'] = nefd
    result['mdcf'] = flim

    return result


def response(result, integrals):
    """
    Calculate the instrument response.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    integrals : dict
         Collection of various background integrals.

    Returns
    -------
    resp : float
        Instrumental response in wavelength units.
    respnu : float
        Instrument response in frequency units.

    """
    c = const.c.to(u.um / u.s).value
    Jy2W = 1e-26  # Convert Jy to W/m2/Hz
    resp = integrals['1']
    respnu = resp * c * Jy2W / (result['lambda_pivot'] ** 2 * 1e3)
    return resp, respnu


def source_descriptions(result, integrals, ffrac):
    """
    Calculate various optical properties of the source.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    integrals : dict
        Collection of various background integrals.
    ffrac : float
        Fraction of total flux in optimal extraction aperture.

    Returns
    -------
    result : dict
        Same as input `results` but with 'source_size'
        (size of the source in arcsec), 'source_fwhm'
        (FWHM of the source), and 'source_rate' (flux
        from source in extraction aperture) populated.

    """
    srate = integrals['4']

    result['source_size'] = integrals['8'] / integrals['4']
    result['source_fwhm'] = integrals['9'] / integrals['4']

    # Energy/s (Watts) from source in extraction aperture
    result['source_rate'] = srate * ffrac
    return result


def color_terms(result, fref, pl, bb, alpha=None, wref=None,
                temp=None, model_flux_in_filter=None, wavelengths=None):
    """
    Calculate the color terms k0 and k1.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    fref : float
        Reference flux for power law models.
    pl : bool
        If set, the model is based on a power law.
    bb : bool
        If set, the model is based on a blackbody.
    alpha : float, optional
        Slope used for power law if `pl` is True.
    wref : float, optional
        Reference wavelength for power law models if `pl` is True.
    temp : float, optional
        Reference temperature for blackbody models if `bb` is True.
    model_flux_in_filter : numpy.array, optional
        Flux in the filter of the model if `pl` and `bb` are
        both False.
    wavelengths : numpy.array, optional
        Wavelenths in the filter of the model if `pl` and `bb` are
        both False.

    Returns
    -------
    result : dict
        Same as input `result` with both 'color_term_k0' and
        'color_term_k1' populated.

    """

    flam_lam0, flam_lam1 = flux_reference_wavelength(result, pl, bb, alpha,
                                                     wref, fref, temp,
                                                     model_flux_in_filter,
                                                     wavelengths)
    k0 = result['flux_mean'] / flam_lam0
    k1 = (result['flux_mean'] * result['lambda_mean']
          / (flam_lam1 * result['lambda_1']))

    result['color_term_k0'] = k0
    result['color_term_k1'] = k1

    return result


def flux_reference_wavelength(result, pl=False, bb=False, alpha=None,
                              wref=None, fref=None,
                              temp=None, model_flux_in_filter=None,
                              wavelengths=None):
    """
    Calculate the flux from the model at reference wavelengths.

    Parameters
    ----------
    result : dict
        Collection of calibration results.
    pl : bool, optional
        If set, the model is based on a power law. Defaults
        to False.
    bb : bool, optional
        If set, the model is based on a blackbody. Defaults
        to False.
    alpha : float, optional
        Slope used for power law if `pl` is True.
    wref : float, optional
        Reference wavelength for power law models if `pl` is True.
    fref : float, optional
        Reference frequency for power law models.
    temp : float, optional
        Reference temperature for blackbody models if `bb` is True.
    model_flux_in_filter : numpy.array, optional
        Flux in the filter of the model if `pl` and `bb` are
        both False.
    wavelengths : numpy.array, optional
        Wavelenths in the filter of the model if `pl` and `bb` are
        both False.

    Returns
    -------
    flam_lam0 : float
        Model flux at the mean wavelength.
    flam_lam1 : float
        Model flux at lambda 1.

    """
    c = const.c.to(u.um / u.s).value
    Jy2W = 1e-26  # Convert Jy to W/m2/Hz

    if pl:
        flam_lam0 = (fref * Jy2W * c
                     * ((wref / result['lambda_mean']) ** alpha)
                     / result['lambda_mean'] ** 2)
        flam_lam1 = (fref * Jy2W * c
                     * ((wref / result['lambda_1']) ** alpha)
                     / result['lambda_1'] ** 2)
    elif bb:
        flam_lam0 = \
            np.pi * thermast.planck_function(result['lambda_mean'], temp)
        flam_lam1 = \
            np.pi * thermast.planck_function(result['lambda_1'], temp)
    else:
        flam_lam0 = iso.interpol(model_flux_in_filter, wavelengths,
                                 result['lambda_mean'])
        flam_lam1 = iso.interpol(model_flux_in_filter, wavelengths,
                                 result['lambda_1'])
    return flam_lam0, flam_lam1


def pointing_optics_sigma(iq):
    """
    Calculate the quality of the pointing optics.

    Parameters
    ----------
    iq : float
        Image quality. 80% enclosed energy giam for
        pointing + optics  in arcsec.

    Returns
    -------
    sig_pt_opt : float
        Radial RMS vale of a 2-D Gaussian corresponding
        to `iq`.

    """
    # d_80 = 2.54*sigma for 80% enclosed energy
    # sig_pt_opt is the radial RMS value (not the sigma) of
    # a 2-D Gaussian
    d_80 = iq
    sig_pt_opt = d_80 / 2.54
    return sig_pt_opt


def source_size(warr, iq, theta_pix):
    """
    Compute the source size.

    Parameters
    ----------
    warr : numpy.array
        Wavelengths in this filter in microns.
    iq : float
        Image quality d_80
    theta_pix : numpy.array
        Pixel size in arcsec.

    Returns
    -------
    fwhm : numpy.array
        FWHM of each pixel in arcsec.
    num_pixels : numpy.array
        Number of pixels in optimal extraction area.
    ffrac : float
        Fraction of total flux in optimal extraction aperture.

    Notes
    -----
    Other variable definitions:
    sig_d : size of diffraction limited beam in arc

    """
    r2a = 206265.  # radians to arcsecs
    Dtel = 2.50e6  # Telescope diameter in um

    sig_pt_opt = pointing_optics_sigma(iq)

    # Size of diffraction limited beam
    sig_d = 0.612 * warr * r2a / Dtel

    # RMS radial size
    r_rms = np.sqrt(sig_pt_opt ** 2 + sig_d ** 2)
    fwhm = 2. * np.sqrt(np.log(2.)) * r_rms

    # Optimal extraction size
    ext = np.pi * 1.121 ** 2 * r_rms ** 2

    num_pixels = ext / theta_pix ** 2
    ffrac = 0.715

    return fwhm, num_pixels, ffrac


def apply_filter(caldata, filter_name, atmosphere_wave,
                 atmosphere_transmission, model_wave, model_flux):
    """
    Only select out wavelength and fluxes in a given filter.

    Parameters
    ----------
    caldata : str
        Path to location of calibration data.
    filter_name : str
        Name of the filter to apply.
    atmosphere_wave : numpy.array
        Wavelengths of the atmosphere transmission model.
    atmosphere_transmission : numpy.array
        Transmission of the atmosphere at each wavelength
        in `atmosphere_wave`.
    model_wave : numpy.array
        Wavelengths of the source model.
    model_flux : numpy.array
        Modeled flux emitted by source at each wavelength
        in `model_wave`.

    Returns
    -------
    wf : numpy.array
        Wavelengths of the filter.
    tf : numpy.array
        Transmission of the filter at each wavelength
        in `wf`.
    taf : numpy.array
        Atmospheric transmission at each wavelength in the filter.
    fsi : numpy.array
        Flux emitted by the modeled source at each wavelength
        in the filter.
    warr : numpy.array
        Wavelengths in the filter that `fsi` is defined at.
    fname : str
        Full path of the filter.

    """
    # Load filter transmission profile
    fname = os.path.join(caldata, filter_name)
    wf, tf = np.loadtxt(fname, skiprows=2,
                        usecols=(0, 1), unpack=True)

    a_indicies = (atmosphere_wave >= min(wf)) & (atmosphere_wave <= max(wf))
    s_indicies = (model_wave >= min(wf)) & (model_wave <= max(wf))
    if a_indicies.sum() > s_indicies.sum():
        warr = atmosphere_wave[a_indicies]
        taf = atmosphere_transmission[a_indicies]
        fsi = iso.interpol(model_flux, model_wave, warr)
    else:
        warr = model_wave[s_indicies]
        taf = iso.interpol(atmosphere_transmission, atmosphere_wave, warr)
        fsi = model_flux[s_indicies]
    return wf, tf, taf, fsi, warr, fname
