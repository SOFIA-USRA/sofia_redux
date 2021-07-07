# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime
import os

import astropy.io.fits as pf
from astropy.modeling import models
import numpy as np
import pandas as pd

import sofia_redux.calibration
import sofia_redux.calibration.standard_model.genastmodel2 as gm


def asteroids():
    location = os.path.dirname(sofia_redux.calibration.__file__)
    asteroid_file = os.path.join(location, 'data', 'models',
                                 'asteroids.dat')
    asteroid = pd.read_csv(asteroid_file, names=['number', 'name'],
                           skiprows=1)
    return asteroid


def horizon():
    param = {'julian': 2458024.9641203703,
             'albedo': 0.09,
             'gmag': 0.12,
             'radius': 469.7,
             'delta': 2.90732021735545,
             'r': 2.628811660414,
             'phi': 20.0506}
    return param


def timepoint():
    timepoint = datetime.datetime(year=2017, month=9, day=28,
                                  hour=11, minute=8, second=20)
    return timepoint


def stm(save=True, temp_dir=None):
    horizons = horizon()
    tp = timepoint()
    model = gm.asteroid_model(params=horizons, date=tp.date(), time=tp.time(),
                              return_model=True, save_model=False)
    model = pd.DataFrame(model.T, columns=['wave', 'flux'])
    if save:
        filename = 'stm_model.out'
        p = temp_dir / filename
        model.to_csv(p, index=False)
    return model


def stm_file(temp_dir):
    model = stm(save=False)
    filename = 'temp_stm_model.csv'
    p = temp_dir / filename
    model.to_csv(p, index=False, sep=' ', header=None)
    return p.strpath


def atran(tmpdir, num_points=100):

    # build header
    header = pf.header.Header()
    header['naxis'] = 2
    header['naxis1'] = num_points
    header['naxis2'] = 2
    header['date'] = '2016-06-02 18:55:57.957061'
    header['model'] = 'ATRAN'
    header['altitude'] = 41000.0
    header['za'] = 45.0
    header['pmv'] = 'default'
    header['xunits'] = 'microns'
    header['yunits'] = ''
    header['xtitle'] = '!7k!5 (!7l!5m)'
    header['ytitle'] = 'Transmission'

    # build data
    wave_min = 40
    wave_max = 300

    wave = np.linspace(wave_min, wave_max, num_points)
    flux = np.ones_like(wave)

    noise = np.random.normal(0, 0.01, num_points)
    flux += noise

    nlines = 10
    means = np.random.randint(wave_min, wave_max, nlines)
    widths = np.random.random(nlines) * 5
    amplitudes = np.random.random(nlines)
    for mean, width, amplitude in zip(means, widths, amplitudes):
        g = models.Gaussian1D(amplitude=amplitude, mean=mean, stddev=width)
        flux -= g(wave)
    flux[flux < 0] = 0
    flux[flux > 1] = 1

    data = np.stack((wave, flux))

    primary = pf.PrimaryHDU(data=data, header=header)
    hdul = pf.HDUList([primary])
    filepath = os.path.join(tmpdir, 'atran_41K_45deg_40-300mum.fits')
    hdul.writeto(filepath, overwrite=True)

    return filepath


def forcast_filters():
    filename = os.path.join(os.path.dirname(__file__), 'data',
                            'forcast_filters.csv')
    filters = pd.read_csv(filename)

    filters['lambda_min'] = filters['lambda_eff'] - filters['lambda_width']
    filters['lambda_max'] = filters['lambda_eff'] + filters['lambda_width']

    return filters


def select_spectel(lamref, filters):
    index = (filters['lambda_min'] < lamref) & (lamref < filters['lambda_max'])
    valid_filters = filters[index]
    spectel = valid_filters.sample(n=1)['SPECTEL'].values[0]
    return spectel


def generate_lamrefs(num, filters):
    np.random.seed(131)
    spectels = filters['SPECTEL'].sample(n=num, replace=True)

    waves = list()
    for spectel in spectels:
        filt = filters[filters['SPECTEL'] == spectel]
        wave_range = filt['lambda_max'] - filt['lambda_min']
        wave = np.random.random() * wave_range + filt['lambda_min']
        waves.append(wave.values[0])
    return waves, spectels
