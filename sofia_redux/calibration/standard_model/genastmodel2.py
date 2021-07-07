# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Generate thermal models for asteroids."""

import os
import argparse
import numpy as np
from scipy.interpolate import interp1d

from sofia_redux.calibration.standard_model import thermast
from sofia_redux.calibration.pipecal_error import PipeCalError


def asteroid_model(params, date, time, outprefix=None,
                   outfile=None, return_model=False, save_model=True):
    """
    Generate a model of the flux from an asteroid.

    Parameters
    ----------
    params : dict
        Contains information about the target pulled from
        JPL's Horizons database.
    date : str, datetime.date
        The date of the observation.
    time : str, datetime.time
        The time of the observation.
    outprefix : str, optional
        A prefix to attached to the beginning of the filename
        created with the model. Defaults to 'test_obs'.
    outfile : str, optional
        Name of output file created with the model. Defaults to
        ' `outprefix` _ `date` _ `time` _model.out'.
    return_model : bool, optional
        If set, the model will be returned. Defaults to False.
    save_model : bool, optional
        If set, write the model to file. Defaults to True.

    Returns
    -------
    model : numpy.array
        An array with two columns. The first is the wavelength
        of the model (um), and the second is the asteroid's
        flux (Jy).

    """
    # Compute thermal models for each input date and
    # interpolate to requested filter wavelengths
    nw = 100000
    warr, fnu = \
        thermast.thermast(params['gmag'], params['albedo'],
                          params['radius'], params['phi'],
                          params['r'], params['delta'],
                          nw=nw)
    model = np.array([warr, fnu])

    fit = interpolate_model(warr, fnu)

    if save_model:
        if not outfile:
            if not outprefix:
                outprefix = 'test_out'
            outfile = f'{outprefix}{date}_{time}_model.out'
            outfile = outfile.replace(':', '')
        write_models(model, fit, params, outfile)

    if return_model:
        return model


def interpolate_model(model_wave, model_flux, requested_wave=None):
    """
    Interpolate the thermal model at key filter wavelengths.

    Parameters
    ----------
    model_wave : numpy.array
        Wavelengths of the thermal model.
    model_flux : numpy.array
        Flux of the themeral model.
    requested_wave : numpy.array, optional
        Wavelengths to interpolate the model at. If not
        provided, the model will be interpolated at
        wavelengths representative of FORCAST and
        HAWC filters.

    Returns
    -------
    fit : numpy.array
        Value of the thermal model at the FORCAST/HAWC
        filter wavelengths.

    """
    # Interpolate the returned function on the filter wavelengths
    if requested_wave is not None:
        wfilt = requested_wave
    else:
        wfilt = [5.356, 6.348, 6.614, 7.702, 8.605, 11.088, 11.342,
                 11.796, 19.670, 24.919, 25.242, 31.383, 33.431,
                 34.678, 37.112, 53.5605, 63.2529, 89.3053,
                 156.480, 216.969]
    flux_function = interp1d(model_wave, model_flux)
    fluxes = flux_function(wfilt)
    fit = np.array([wfilt, fluxes])
    return fit


def write_models(model, fit, params, outfile):
    """
    Write all models to file.

    Parameters
    ----------
    model : numpy.array
        Thermal model of asteroid flux.
    fit : numpy.array
        Interpolated fit of `model`. Written to a file
        called 'bb_fit.out'.
    params : dict
        Output of call to JPL Horizons database.
    outfile : str
        Name of file to write `model` to.

    Returns
    -------
    None

    """
    header = (f'STM parameters: \n'
              f'\tGmag = {params["gmag"]}\n'
              f'\tAlbedo = {params["albedo"]}\n'
              f'\tRadius = {params["radius"]} km\n'
              f'\tPhase = {params["phi"]} deg\n'
              f'\tD_sun = {params["r"]} AU\n'
              f'\tD_earth = {params["delta"]} AU\n'
              f'Wavelength\tFlux\n'
              f'(microns)\t(Jy)')
    np.savetxt(outfile, model.T, header=header, comments='; ', fmt='%.6f')
    np.savetxt('bb_fit.out', fit.T)


def parse_args(args):
    """
    Parse the command line arguments.

    Parameters
    ----------
    args : list
        Arguments passed in on the command line.

    Returns
    -------
    args : argparse.Namespace
        Namespace populated with the options passed in.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('params', metavar='params',
                        type=str, nargs=1,
                        help='Output from JPL Horizons database')
    parser.add_argument('obs_file', metavar='obs_file',
                        type=str, nargs=1,
                        help='File containing list of observation '
                             'dates and times')
    parser.add_argument('outprefix', metavar='outprefix',
                        type=str, default='',
                        help='Prefix for output file')

    args = parser.parse_args(args)
    return args


def check_args(args):
    """
    Check the command line arguments are valid.

    Parameters
    ----------
    args : argsparse.Namespace
        Namespace populated with the options passed in.

    Returns
    -------
    args : argparse.Namespace
        Same as input `args` with default values checked.

    Raises
    ------
    PipeCalError : exception
        Raised if any of the input files do not exist.

    """
    if isinstance(args.params, list):
        args.params = args.params[0]
    if not os.path.isfile(args.params):
        raise PipeCalError(f'Value provided for param argument '
                           f'{args.params} does not exist.')

    if isinstance(args.obs_file, list):
        args.obs_file = args.obs_file[0]
    if not os.path.isfile(args.obs_file):
        raise PipeCalError(f'Value provided for obs_file argument '
                           f'{args.obs_file} does not exist.')

    if not args.outprefix:
        args.outprefix = 'test_out'

    return args
