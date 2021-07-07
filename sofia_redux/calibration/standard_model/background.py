# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Derive flux expected from background sources"""

import numpy as np
import pandas as pd
import scipy.integrate as si
import astropy.constants as const
import astropy.units as u

from sofia_redux.calibration.standard_model import thermast


def derive_background_photon_flux(warr, temperatures):
    """
    Setup planck functions for each non-target source.

    Typical sources are the telescope, foreoptics,
    atmosphere, window, and instrument.

    Parameters
    ----------
    warr : numpy.array
        Wavelengths to determine the blackbody flux at, in microns.
    temperatures : dict
        Sources to generate blackbodies for. The keys are the
        name of the source, and values are the temperatures
        of each source.

    Returns
    -------
    plancks : dict
        Collection of blackbody spectrums for each source in
        `temperatures`. The keys are the same as `temperature` and
        the values are numpy arrays holding the blackbody flux.

    """
    plancks = dict()
    for source, temperature in temperatures.items():
        plancks[source] = thermast.planck_function(warr, temperature)
    return plancks


def background_integrand_1(plancks, emissivity, etas, total_throughput, tfi,
                           filter_number):
    """
    Calculate the first background integral.

    Parameters
    ----------
    plancks : dict
        Collection of background sources and their corresponding
        blackbody emission.
    emissivity : dict
        Collection of background sources and their corresponding
        emissivity.
    etas : dict
        Collection of background sources and their corresponding
        transmissions.
    total_throughput : numpy.array
        Total throughput (telescope + filter + instrument) as a
        function of wavelength.
    filter_number : int
        Current filter number.

    Returns
    -------
    final : numpy.array
        Integration of the product of all throughputs as a
        function of wavelength.

    """
    throughputs = dict()
    throughputs['atmosphere'] = total_throughput
    throughputs['telescope'] = total_throughput / etas['telescope']
    throughputs['foreoptics'] = (tfi * etas['window'][filter_number]
                                 * etas['instrument'][filter_number])
    throughputs['window'] = etas['instrument'][filter_number] * tfi
    throughputs['instrument'] = etas['instrument'][filter_number]

    emiss = dict()
    emiss['atmosphere'] = emissivity['atmosphere']
    emiss['telescope'] = emissivity['telescope']
    emiss['foreoptics'] = emissivity['foreoptics']
    emiss['window'] = emissivity['window'][filter_number]
    emiss['instrument'] = 1

    integrand = pd.DataFrame({'plancks': plancks, 'emissivity': emiss,
                              'throughput': throughputs})

    integrand['product'] = integrand.prod(axis=1)
    final = integrand['product'].sum()

    return final


def background_integrand_2(plancks, temperatures, warr, total_throughput,
                           emissivity, etas, transmissions, filter_number):
    """
    Calculate the second background integral.

    Parameters
    ----------
    plancks : dict
        Collection of background sources and their corresponding
        blackbody emission.
    temperatures : dict
        Collection of background sources and their corresponding
        temperatures.
    warr : numpy.array
        Wavelengths to integrate over, in microns.
    total_throughput : numpy.array
        Total throughput (telescope + filter + instrument) as a
        function of wavelength.
    emissivity : dict
        Collection of background sources and their corresponding
        emissivity.
    etas : dict
        Collection of background sources and their corresponding
        transmissions.
    transmissions : numpy.array
        Transmission of filters.
    filter_number : int
        Current filter number.

    Returns
    -------
    final : float
        Evaluation of integrating everything over wavelength.

    """
    field_order = ['atmosphere', 'telescope', 'window',
                   'foreoptics', 'instrument']
    numerators = dict()
    numerators['atmosphere'] = [emissivity['atmosphere'], total_throughput]
    numerators['telescope'] = [emissivity['telescope'],
                               total_throughput / etas['telescope']]
    numerators['foreoptics'] = [emissivity['foreoptics'],
                                etas['window'][filter_number],
                                etas['instrument'][filter_number],
                                transmissions]
    numerators['window'] = [emissivity['window'][filter_number],
                            etas['instrument'][filter_number],
                            transmissions]
    numerators['instrument'] = [etas['instrument'][filter_number]]
    temperatures = temperatures.copy()
    temperatures['instrument'] = temperatures['window']
    etas['atmosphere'] = 0
    factors = integrand2(numerators, warr, temperatures,
                         field_order=field_order)

    integrand = pd.DataFrame({'plancks': plancks, 'factors': factors})
    final = integrand.prod(axis=1).sum()

    return final


def integrand2(numerators, warr, temperatures, field_order):
    """
    Calculate scale factors for terms in background_intetgral_2.

    Parameters
    ----------
    numerators : dict
        Collection of sources and their corresponding
        effective emissivities in the current filter.
    warr : numpy.array
        Wavelengths to integrate over, in microns.
    temperatures : dict
        Collection of sources and their corresponding
        temperatures.
    field_order : list
        List of the keys in `numerators` and `temperatures`
        to ensure everything is done in the right order.

    Returns
    -------
    factors : dict
        Background integral coeffecient for each source.

    """
    factors = dict()
    for i, field in enumerate(field_order):
        numerator = numerators[field]
        temperature = temperatures[field]
        factor = background_integrand_coeff(numerator, warr, temperature)
        factors[field] = factor
    return factors


def background_integrand_coeff(numerator, warr, temperature):
    """
    Coefficient of the background integral for each source.

    Parameters
    ----------
    numerator : list
        Collection of various emissivity/throughput for the given source.
    warr : numpy.array
        Wavelengths to integrate over, in microns.
    temperature : float
        Temperature of the given source.

    Returns
    -------
    coeff : numpy.array
        Background integral coefficient.

    """
    exponential_factor = (const.h * const.c / const.k_B).to(u.um * u.K).value
    top = np.prod(numerator, axis=0)**2
    bottom = np.exp(exponential_factor / (warr * temperature)) - 1
    return top / bottom


def background_power(telescope_area, omega_pix, bg_integrand1,
                     num_pix, wavelengths):
    """
    Background power in the extraction area.

    Parameters
    ----------
    telescope_area : float
        Area of the telescope in microns^2.
    omega_pix : float
        Pixel sold angle in steradians.
    bg_integrand1 : numpy.array
        Background integrand at each eavelength.
    num_pix : numpy.array
        Number of pixels in the extraction area for each wavelength.
    wavelengths : numpy.array
        Wavelength to integrate over, in microns.

    Returns
    -------
    power : float
        Background power, in Watts.

    """

    integral = si.simps(bg_integrand1 * num_pix, wavelengths)
    power = telescope_area * omega_pix * integral
    return power


def setup_integrands(total_throughput, atmosphere_transmission, telescope_area,
                     warr, model_flux, num_pix, fwhm):
    """
    Define all the integrands to be used.

    Parameters
    ----------
    total_throughput : numpy.array
        Total throughput at each wavelength.
    atmosphere_transmission : numpy.array
        Atmosphere transmission as a function of wavelength.
    telescope_area : float
        Area of telescope in microns^2.
    warr : numpy.array
        Wavelengths to integrate over, in microns.
    model_flux : numpy.array
        Modelled flux of the source as a function of wavelength.
    num_pix : numpy.array
        Number of pixels in the extraction area as a function
        of wavelength.
    fwhm : numpy.array
        FWHM of source as a function of wavelength.

    Returns
    -------
    integrands : dict
        The required integrands for fully characterizing the
        background flux.

    """
    integrands = dict()
    integrands['0'] = total_throughput * atmosphere_transmission
    integrands['1'] = integrands['0'] * telescope_area
    integrands['2'] = integrands['1'] * warr
    integrands['3'] = integrands['1'] / warr
    integrands['4'] = integrands['1'] * model_flux
    integrands['5'] = integrands['4'] * warr
    integrands['6'] = integrands['5'] * warr
    integrands['7'] = integrands['3'] * np.log(warr)
    integrands['8'] = integrands['4'] * num_pix
    integrands['9'] = integrands['4'] * fwhm
    integrands['10'] = integrands['2'] * warr
    integrands['11'] = integrands['3'] / warr
    integrands['12'] = integrands['11'] / warr

    return integrands


def integrate_integrands(integrands, warr):
    """
    Integrate the functions set up in `setup_integrands`.

    Parameters
    ----------
    integrands : dict
        Collection of integrands.
    warr : numpy.array
        Wavlengths to integrate over.

    Returns
    -------
    integrals : dict
        The integrals of each integrand in `integrands`
        over wavelength.

    """
    integrals = dict()
    for key, integrand in integrands.items():
        integrals[key] = si.simps(integrand, warr)
    return integrals
