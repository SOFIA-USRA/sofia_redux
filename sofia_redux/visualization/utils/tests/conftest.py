#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
from astropy.modeling import models
import matplotlib.axis as mpa
import matplotlib.figure as mpf


@pytest.fixture(scope='function')
def lorentz_fit():
    moffat = models.Lorentz1D(amplitude=2, x_0=4, fwhm=0.1)
    line = models.Linear1D(slope=0.5, intercept=2)
    return moffat + line


@pytest.fixture(scope='function')
def moffat_fit():
    moffat = models.Moffat1D(amplitude=2,
                             x_0=5,
                             gamma=3,
                             alpha=0.2)
    line = models.Linear1D(slope=0.5, intercept=2)
    return moffat + line


@pytest.fixture(scope='function')
def gauss_fit():
    gauss = models.Gaussian1D(amplitude=1, mean=0, stddev=0.2)
    line = models.Linear1D(slope=0.5, intercept=2)
    return gauss + line


@pytest.fixture(scope='function')
def gauss_const_fit():
    gauss = models.Gaussian1D(amplitude=1, mean=0, stddev=0.2)
    const = models.Const1D(amplitude=2.)
    return gauss + const


@pytest.fixture(scope='function')
def gauss_params(gauss_fit):
    params = {'fit': gauss_fit,
              'axis': mpf.Figure().subplots(1, 1),
              'status': 'pass',
              'x_field': 'wavelength', 'x_unit': 'nm',
              'y_field': 'flux', 'y_unit': 'Jy',
              'visible': False, 'lower_limit': 5,
              'upper_limit': 15}
    return {'model_id_name': {1: params}}


@pytest.fixture(scope='function')
def single_gauss_fit():
    gauss = models.Gaussian1D(1, 0, 0.2)
    return gauss


@pytest.fixture(scope='function')
def single_gauss_params(single_gauss_fit):
    params = {'fit': single_gauss_fit,
              'axis': mpf.Figure().subplots(1, 1),
              'status': 'pass',
              'x_field': 'wavelength', 'x_unit': 'nm',
              'y_field': 'flux', 'y_unit': 'Jy',
              'visible': False, 'lower_limit': 5,
              'upper_limit': 15}
    return {'model_id_name': {1: params}}


