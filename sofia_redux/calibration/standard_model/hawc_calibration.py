# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Generates models of calibration objects for HAWC+ observations"""

import datetime as dt
import argparse
import sys
import os
import pandas as pd

from sofia_redux.calibration.standard_model \
    import horizons, genastmodel2, hawc_calib, modconvert
from sofia_redux.calibration.pipecal_error import PipeCalError


def calibration_data_path():
    """
    Determine the location of calibration data.

    Returns
    -------
    caldata : str
        Absolute path of the calibration directory.

    """
    # get the calibration data path
    pkgpath = (os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) + os.path.sep)
    caldata = os.path.join(*[pkgpath, 'data', 'models'])
    return caldata


def model_dist(target, caldata):
    """
    Determine the distance to a target in a Herschel model.

    Parses the herschel readme to get the distance of
    the object used in the Herschel model.

    Parameters
    ----------
    target : str
        Name of the target.
    caldata : str
        Path to the calibration data.

    Returns
    -------
    d_model : float
        Distance to the target in AU.

    """
    # Get the distance of the model:
    d_model = None
    herschel_parameters = os.path.join(caldata, 'readme.txt')
    with open(herschel_parameters) as f:
        for line in f:
            if line.lower().startswith(target.lower()):
                d_model = float(line.split()[3])
                d_model *= 6.68459e-9
    if not d_model:
        raise PipeCalError(f'Unable to find major target {target} in '
                           f'{herschel_parameters}')
    return d_model


def generate_major_outfile(target, herschel_file, date):
    """
    Generate the name of the output file for major targets.

    Parameters
    ----------
    target : str
        Name of the target.
    herschel_file : str
        Name of the Herschel file for `target`.
    date : str
        Date of the observation.

    Returns
    -------
    model_outfile : str
        Name of output file to write to.

    """
    esa = os.path.basename(herschel_file).split('_')[1].upper()
    date_clean = dt.datetime.strptime(date, '%Y-%m-%d').strftime('%Y%b%d')
    model_outfile = f'{target.capitalize()}_{date_clean}_{esa}.txt'
    return model_outfile


def apply_scale_factor(herschel_file, outfile, fscale):
    """
    Apply a scale factor to Herschel model to account for distance.

    Parameters
    ----------
    herschel_file : str
        Name of file containing Herschel model.
    outfile : str
        Name of file to write scaled model to.
    fscale : float
        Scale to apply.

    Returns
    -------
    model : pandas.DataFrame
        Scaled Herschel model.

    """
    model = modconvert.modconvert(herschel_file, outfile, fscale)
    return model


def scale_factor(distance, params):
    """
    Determine the scale factor for accounting for distance to target.

    Parameters
    ----------
    distance : float
        Distance to the target in AU.
    params : dict
        Parameters describing the Herschel model.

    Returns
    -------
    fscale : float
        Scale factor to apply to model flux.

    """
    fscale = (distance / params['delta']) ** 2
    return fscale


def classify_target(target):
    """
    Determine if a target is a minor or major target.

    Parameters
    ----------
    target : str
        Target name.

    Returns
    -------
    classification : str
        Set to 'major' if target is Uranus, Neptune, Ganymede, or
        Callisto. Set to 'minor' otherwise.

    """

    major_targets = ['uranus', 'neptune', 'ganymede', 'callisto']
    if target.lower() in major_targets:
        return 'major'
    else:
        return 'minor'


def read_obstimes(obs_files):
    """
    Read in when the observations took place.

    Parameters
    ----------
    obs_files : str
        Name of the file containing the description of the
        observations.

    Returns
    -------
    obs_times : pandas.DataFrame
        Details of the observations.

    Raises
    ------
    PipeCalError
        If any part of `obs_files` is imporoperly formatted.

    """
    try:
        obs_times = pd.read_csv(obs_files, names=['date', 'time', 'target'],
                                sep=r'\s+')
    except IOError:
        raise PipeCalError(f'Unable to read obs_times file {obs_files}')
    if obs_times.isna().sum().sum() > 0:
        raise PipeCalError(f'Obs_times file {obs_files} is improperly '
                           f'formatted.')
    obs_times['datetime'] = obs_times['date'] + 'T' + obs_times['time']
    try:
        obs_times['datetime'] = pd.to_datetime(obs_times['datetime'])
    except (SyntaxError, ValueError):
        raise PipeCalError(f'Obs_times file {obs_files} has improperly '
                           f'formatted dates/times.')

    return obs_times


def select_herschel_file(target):
    """
    Determine the correct Herschel model to use for the target.

    Parameters
    ----------
    target : str
        Name of target.

    Returns
    -------
    herschel_model : str
        Absolute path to the file containing the correct
        Herschel model.

    """
    caldata = calibration_data_path()
    herschel_files = {'uranus': 'ura_esa2_2_i.dat',
                      'neptune': 'nep_esa5_2_i.dat',
                      'ganymede': 'gany_esa2_2_i.dat',
                      'callisto': 'call_esa2_2_i.dat'}
    herschel_model = os.path.join(caldata, herschel_files[target.lower()])
    return herschel_model


def parse_atran_filename(atran):
    """
    Pull the alittude and zenith angle from the ATRAN filename.

    Parameters
    ----------
    atran : str
        Name of the ATRAN file.

    Returns
    -------
    altitude : str
        Altitude of the plane for the atmospheric model.
    zenith : str
        Zenith angle for the atmospheric model.

    """
    altitude = os.path.basename(atran).split('_')[1]
    zenith = os.path.basename(atran).split('_')[2]
    return altitude, zenith


def generate_major_cal_outfile(row, atran, herschel_file):
    """
    Generate the name of the output file for a major target.

    Parameters
    ----------
    row : pandas.Series
        Details of a single observation.
    atran : str
        Name of the ATRAN file used.
    herschel_file : str
        Name of the standard Herschel file for
        this target.

    Returns
    -------
    calib_outfile : str
        Name of the calibration outfile to write the model to.

    """
    date = row['date']
    if isinstance(date, str):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    date_clean = date.strftime('%Y%b%d')
    esa = os.path.basename(herschel_file).split('_')[1].upper()
    altitude, zenith = parse_atran_filename(atran)
    calib_outfile = (f'HAWC_'
                     f'{row["target"].capitalize()}_{esa}_'
                     f'{date_clean}_{altitude}_'
                     f'{zenith}.out')
    return calib_outfile


def model_minor_body(obs_times, atran):
    """
    Generate a model for a minor target.

    Parameters
    ----------
    obs_times : pandas.DataFrame
        Inforamtion describing all observations of the target.
    atran : str
        Name of ATRAN file to use for atmospheric modeling.

    Returns
    -------
    obs : pandas.DataFrame
        Copy of `obs_times` with added columns describing the
        name of the model and calibration files.

    """
    obs = pd.DataFrame()
    for index, row in obs_times.iterrows():
        params = horizons.asteroid_query(
            target=row['target'], date=row['date'], time=row['time'])
        model_outfile = generate_minor_outfile(row)
        calibration_outfile = generate_minor_cal_outfile(row, atran)
        genastmodel2.asteroid_model(params=params, date=row['date'],
                                    time=row['time'], outfile=model_outfile)
        row['model_file'] = model_outfile
        row['cal_file'] = calibration_outfile
        obs = obs.append(row)
    return obs


def generate_minor_outfile(row):
    """
    Generate the name of the output file for a minor target.

    Parameters
    ----------
    row : pandas.Series
        Details of a single observation.

    Returns
    -------
    outfile : str
        Name of the outfile to write the model to.

    """
    target = row['target']
    date = row['date']
    time = row['time']
    outfile = f'{target.capitalize()}_{date}_{time}_model.out'
    outfile = outfile.replace(':', '')
    return outfile


def generate_minor_cal_outfile(row, atran):
    """
    Generate the name of the output file for a minor target.

    Parameters
    ----------
    row : pandas.Series
        Details of a single observation.
    atran : str
        Name of ATRAN file used for atmosphere modeling.

    Returns
    -------
    calib_outfile : str
        Name of the calibration outfile to write the model to.

    """
    date = row['date']
    if isinstance(date, str):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    date_clean = date.strftime('%Y%b%d')
    target = row['target'].capitalize()
    altitude, zenith = parse_atran_filename(atran)
    calib_outfile = (f'HAWC_{target.capitalize()}_{date_clean}_{altitude}_'
                     f'{zenith}.out')
    return calib_outfile


def model_major_body(obs_times, caldata, atran):
    """
    Generate a model for a major target.

    Parameters
    ----------
    obs_times : pandas.DataFrame
        Information describing all observations of the target.
    caldata : str
        Path to the calibration data.
    atran : str
        Name of ATRAN file to use for atmospheric modeling.

    Returns
    -------
    obs : pandas.DataFrame
        Copy of `obs_times` with added columns describing the
        name of the model and calibration files.

    """
    obs = pd.DataFrame()
    for index, row in obs_times.iterrows():
        herschel_file = select_herschel_file(row['target'])
        model_outfile = \
            generate_major_outfile(row['target'], herschel_file, row['date'])
        calibration_outfile = \
            generate_major_cal_outfile(row, atran, herschel_file)
        params = horizons.simple_query(row['target'], row['date'], row['time'])
        distance = model_dist(row['target'], caldata)
        fscale = scale_factor(distance, params)
        apply_scale_factor(herschel_file, model_outfile, fscale)
        row['model_file'] = model_outfile
        row['cal_file'] = calibration_outfile
        obs = obs.append(row)
    return obs


def calibration(obs_file, atran):
    """
    Generate models of the observed flux from a target.

    Parameters
    ----------
    obs_file : str
        File containing information about all observations.
    atran : str
        Name of the ATRAN file to use for modeling
        atmospheric transmission.

    Returns
    -------
    None

    """

    obs_times = read_obstimes(obs_file)

    caldata = calibration_data_path()

    target_type = classify_target(obs_times['target'][0])

    if target_type == 'major':
        obs_times = model_major_body(obs_times, caldata, atran)
    else:
        obs_times = model_minor_body(obs_times, atran)
    hawc_calib.hawc_calib(obs_times, atran, emiss=0.15, dataframe=True)


def main(args=None):
    """
    Entry point for calibration routines.

    Parameters
    ----------
    args : list, optional
        Any arguments passed in to configure the run.
        If not provided arguements are pulled from
        the command line.

    Returns
    -------
    None

    """
    if args is None:  # pragma: no cover
        args = sys.argv[1:]
    args = parse_args(args)
    args = check_args(args)
    print(f'Obs file: {args.obs_file}')
    print(f'ATRAN: {args.atran}')
    calibration(args.obs_file, args.atran)


def parse_args(args):
    """
    Parse the arguments for configuring calibrations.

    Parameters
    ----------
    args : list
        List of arguments passed in to `main`.

    Returns
    -------
    args : argparse.Namespace
        Arguemnts parsed into an object.

    """
    parser = argparse.ArgumentParser(description='Check FITS headers')
    parser.add_argument('obs_file', metavar='obs_file', type=str, nargs=1,
                        help='file containing dates and times')
    parser.add_argument('-a', '--atran', dest='atran', type=str,
                        action='store', default=None,
                        help='name of ATRAN file')
    args = parser.parse_args(args)
    return args


def check_args(args):
    """
    Verify calibration configuration is valid.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments to configure the run.

    Returns
    -------
    args : argparse.Namespace
        Same as input `args` but with all arguments checked.

    """
    if args.atran is None:
        args.atran = 'atran_41K_45deg_40-300mum.fits'
    if isinstance(args.obs_file, list):
        args.obs_file = args.obs_file[0]
    return args


if __name__ == '__main__':  # pragma: no cover
    args = parse_args(sys.argv[1:])
    main(args)
