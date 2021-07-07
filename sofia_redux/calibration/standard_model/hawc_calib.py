# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Generate a thermal model for a given object at a specified time"""

import numpy as np
import pandas as pd

from sofia_redux.calibration.standard_model import calibration_io
from sofia_redux.calibration.standard_model import background as bg
from sofia_redux.calibration.standard_model import isophotal_wavelength as iso
from sofia_redux.calibration.standard_model import derived_optics as dopt


def hawc_calib(infile, atmofile, outfile=None,
               iq=5.3, no_atm=False, emiss=0.15, snrval=4.0,
               alpha=None, temp=None, normfits=False,
               fscal=1.0, txt=False, noplot=False,
               dataframe=False):
    """
    Calculate source flux from calibrated observations.

    Parameters
    ----------
    infile : str, pandas.DataFrame
        Name of file containing the calibrated model. Typically
        this is a Herschel model, a blackbody model, or a
        power law model. The file can be formatted in a FITS file,
        a pandas DataFrame, or plain ASCII text. If a generic model
        is preferred, `infile` can be set to 'Blackbody' or
        'PowerLaw', to generate a blackbody or power law model
        respectively.
    atmofile : str
        Name of the ATRAN file to use for modelling the
        atmosphere.
    outfile : str, optional
        Name of the file to write the final derived flux.
        If not provided, try to pull the filename from the
        contents of `infile`. If that fails, write the model
        to 'flux_values.out'.
    iq : float, optional
        Image quality d_80. Defaults to 5.3.
    no_atm : bool, optional
        If set, do not take atmosphere contained in `atmofile`
        into account. Defaults to False.
    emiss : float, optional
        Telescope emissivity. Defaults to 0.15.
    snrval : float, optional
        Signal to noise. Defaults to 4.0.
    alpha : float, optional for non-power law models
        If `infile` is `PowerLaw`, `alpha` is the power law
        index to use to generate the model.
    temp : float, optional for non-blackbody models
        If `infile` is 'Blackbody', `temp` is the temperature to
        use to generate the model.
    normfits : bool, optional for non-FITS infiles
        If set, `infile` is a FITS file.
    fscal : float, optional
        Scale factor to apply to fluxes. Defaults to 1.0.
    txt : bool, optional
        If set, `infile` is a plain text ASCII file.
    noplot : bool, optional
        If set, do not create a plot of the results. Defaults
        to False.
    dataframe : bool, optional
        If set, `infile` is a pandas DataFrame.

    Returns
    -------
    None

    Notes
    -----
    Process:
      #. Read in the model of the target flux, which is either
         a scaled Herschel model for major targets or a scaled
         blackbody if an asteroid.
      #. Read in the ATRAN file to get the atmospheric transmission
         as a function of wavelength.
      #. Loop through each HAWC+ filter
      #. Calculate the total transmission (telescope, filter, and
         instrument).
      #. Calculate the size of the source in pixels
      #. Calculate the total background photon flux from the
         sky, telescope, window, foreoptics, and instrument.
      #. Several integrals to calculate everything. See Tokunaga,
         Vacca (2005) for details on what's happening.
      #. Write results to file

    """

    # Constants and converstions
    # h = 6.62607e-27  # ergs-s
    c = 2.99792e10  # cm/s

    cm2mum = 1e4  # Convert cm to microns
    # ergs2W = 1e-7  # Convert ergs/s to Watts
    r2a = 206265.  # radians to arcsecs
    Jy2W = 1e-26  # Convert Jy to W/m2/Hz

    # Telescope diameter in m
    telescope_diameter = 2.50
    # Telescope area in m^2
    telescope_area = np.pi * telescope_diameter ** 2 / 4.

    # Temperatures
    temperatures = setup_temperatures()

    telescope_emissivity = emiss  # Telescope emissivity

    # HAWC parameters
    wmin = 40.0
    wmax = 300.0

    # Pixel size arsec
    theta_pix = np.array([2.6, 4.0, 4.0, 6.9, 9.4])
    # Pixel solid angle steradians
    omega_pix = (theta_pix / r2a) ** 2

    eta_tel = 1 - telescope_emissivity
    eta_fo = 0.96
    eta_wrefl = 0.92
    eta_wabs = np.array([0.63, 0.67, 0.79, 0.86, 0.87])
    eta_win = eta_wabs * eta_wrefl
    eta_inst = np.array([0.146, 0.190, 0.213, 0.286, 0.247])
    tput = eta_tel * eta_fo * eta_win * eta_inst
    etas = {'telescope': eta_tel, 'foreoptics': eta_fo,
            'window': eta_win, 'instrument': eta_inst}

    window_emissivity = 1 - eta_wabs  # Window emissivity
    foreoptics_emissivity = 1 - eta_fo  # Foreoptics emissivity
    emissivity = {'window': window_emissivity,
                  'foreoptics': foreoptics_emissivity,
                  'telescope': telescope_emissivity,
                  'atmosphere': np.nan,
                  'instrument': 1}

    # Filter transmission files
    fnames = [f'HAWC_band{i}.txt' for i in list('ABCDE')]
    filter_names = [s.split('.')[0].split('_')[-1] for s in fnames]
    lam_c = [53.0, 63.0, 89.0, 155.0, 216.0]

    Nf = len(fnames)
    columns = ['lambda_c', 'lambda_mean', 'lambda_1', 'lambda_pivot',
               'lambda_eff', 'lambda_eff_jv', 'width',
               'response', 'flux_mean', 'flux_nu_mean', 'color_term_k0',
               'color_term_k1', 'source_rate', 'source_size', 'source_fwhm',
               'background_power', 'nep', 'nefd', 'mdcf', 'npix_mean',
               'lambda_prime', 'lamcorr', 'isophotal', 'isophotal_wt']

    for index, obs in infile.iterrows():
        print(f"\nObservation: {obs['date']}, {obs['time']}")
        calibration_results = pd.DataFrame(columns=columns, index=filter_names)
        calibration_results.rename_axis(index='filter')

        # Read in stellar spectrum input file
        wstar, fstar, pl, bb = \
            calibration_io.model_spectrum(infile, txt, dataframe,
                                          alpha, temp, df_index=index)

        # Pull out the unique wavelengths
        model_wave, model_flux = \
            calibration_io.unique_wavelengths(wstar, fstar,
                                              wmin=wmin, wmax=wmax)

        # Convert stellar flux to W/m2/micron
        model_flux *= Jy2W * c * cm2mum / (model_wave ** 2)

        atran_group = \
            calibration_io.read_atran(atmofile, model_wave, no_atm=no_atm,
                                      wmin=wmin, wmax=wmax)
        atmosphere_wave, atmosphere_transmission, afile = atran_group

        # Local data path
        caldata = calibration_io.calibration_data_path()

        # Open output file
        outf = calibration_io.open_outfile_and_header(outfile, no_atm, afile,
                                                      infile, index)

        isophotal_weights = np.zeros(len(fnames))

        # Start main loop
        for i in range(0, Nf):

            result = pd.Series(index=columns, name=filter_names[i])
            result['lambda_c'] = lam_c[i]

            filtered_models = \
                dopt.apply_filter(caldata, fnames[i], atmosphere_wave,
                                  atmosphere_transmission, model_wave,
                                  model_flux)
            (filter_wavelength, filter_tranmission,
             atmosphere_transmission_in_filter, model_flux_in_filter,
             wavelengths, filter_name) = filtered_models

            emissivity['atmosphere'] = 1.0 - atmosphere_transmission_in_filter

            # Fit splines to the transmissions
            tfi = iso.interpol(filter_tranmission, filter_wavelength,
                               wavelengths)

            # Total instrument+filter+telescope throughput
            total_throughput = tfi * tput[i]

            fwhm, num_pix, ffrac = dopt.source_size(wavelengths, iq,
                                                    theta_pix[i])

            # Computer background photon flux from various components
            plancks = bg.derive_background_photon_flux(wavelengths,
                                                       temperatures)

            bg_integrand1 = \
                bg.background_integrand_1(plancks, emissivity, etas,
                                          total_throughput, tfi,
                                          filter_number=i)
            bg_integrand2 = \
                bg.background_integrand_2(plancks, temperatures, wavelengths,
                                          total_throughput, emissivity, etas,
                                          transmissions=tfi, filter_number=i)

            # Mean number of pixels in extraction area
            result['npix_mean'] = \
                dopt.mean_pixels_in_beam(num_pix, total_throughput,
                                         wavelengths)

            # Background power W in extraction area (beam)
            result['background_power'] = \
                bg.background_power(telescope_area, omega_pix[i],
                                    bg_integrand1, num_pix, wavelengths)

            result['nep'] = \
                dopt.noise_equivalent_power(bg_integrand1, bg_integrand2,
                                            num_pix, wavelengths, omega_pix[i],
                                            telescope_area)

            # Compute integrals for wavelengths and mean_flux
            integrands = \
                bg.setup_integrands(total_throughput,
                                    atmosphere_transmission_in_filter,
                                    telescope_area, wavelengths,
                                    model_flux_in_filter, num_pix,
                                    fwhm)
            integrals = bg.integrate_integrands(integrands, wavelengths)

            result['width'] = integrals['0']

            # lambdas = calculated_lambdas(integrals)
            result = iso.calculated_lambdas(result, integrals)

            result = dopt.mean_fluxes(result, integrals)

            resp, result['response'] = dopt.response(result, integrals)

            result = dopt.source_descriptions(result, integrals, ffrac)

            result = iso.calc_isophotal(result, wavelengths, pl, alpha,
                                        model_flux_in_filter,
                                        atmosphere_transmission_in_filter,
                                        total_throughput)

            result = dopt.limiting_flux(result, integrals, snrval)

            fref, wref = None, None
            result = dopt.color_terms(result, fref, pl, bb, alpha, wref,
                                      temp, model_flux_in_filter, wavelengths)

            calibration_results.loc[filter_names[i]] = result

            # Print output values
            calibration_io.report_result(result, filter_name, outf)

            calibration_results.to_csv('test_results.csv')

        # Plot fmean values at lambda_iso wavelengths
        if not noplot:
            plot_name = (f"{obs['target'].capitalize()}_{obs['date']}_"
                         f"{obs['time']}_model.out")
            plot_name = plot_name.replace(':', '')
            calibration_io.plot_spectrum(model_wave, model_flux, pl, bb,
                                         isophotal_weights,
                                         calibration_results,
                                         outfile=plot_name)

        # Close files
        outf.close()


def setup_temperatures():
    """
    Define the temperatures of background sources.

    Returns
    -------
    temperatures : dict
        Collection of sources and their corresponding temperatures.

    """
    Ttel = 240.  # Telescope temperature
    Tsky = 240.  # Sky temperature
    Twin = 278.  # Window temperature
    Tfo = 293.  # Foreoptics temperature
    Tinst = 10.  # Internal optics temperature

    temperatures = {'atmosphere': Tsky, 'telescope': Ttel,
                    'window': Twin, 'foreoptics': Tfo,
                    'instrument': Tinst}
    return temperatures
