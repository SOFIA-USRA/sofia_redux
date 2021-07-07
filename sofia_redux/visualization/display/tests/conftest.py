#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.modeling import models
from matplotlib import figure as mpf
import matplotlib.backends.backend_qt5agg as mb
from PyQt5 import QtWidgets

from sofia_redux.visualization.display import (pane, blitting, artists,
                                               figure, view)
from sofia_redux.visualization.display.ui import mplwidget
from sofia_redux.visualization.utils import model_fit
from sofia_redux.visualization import signals


@pytest.fixture(scope='function')
def fig():
    fig = mpf.Figure()
    return fig


@pytest.fixture(scope='function')
def ax(fig):
    ax = fig.add_subplot()
    return ax


@pytest.fixture(scope='function')
def one_dim_pane(ax):
    obj = pane.OneDimPane(ax)
    return obj


@pytest.fixture(scope='function')
def line(one_dim_pane):
    x = np.arange(1, 10, 1)
    y = x + 2
    art = one_dim_pane.ax.plot(x, y)[0]
    return art


@pytest.fixture(scope='function')
def line_alt(one_dim_pane):
    x = np.arange(1, 10, 1)
    y = x + 2
    one_dim_pane.ax_alt = one_dim_pane.ax.twinx()
    art = one_dim_pane.ax_alt.plot(x, y)[0]
    return art


@pytest.fixture(scope='function')
def scatter(one_dim_pane):
    x = np.arange(1, 10, 1)
    y = x + 2
    art = one_dim_pane.ax.scatter(x, y)
    return art


@pytest.fixture(scope='function')
def guide(one_dim_pane):
    art = one_dim_pane.ax.axvline()
    return art


@pytest.fixture(scope='function')
def fit(one_dim_pane):
    x = np.arange(1, 10, 1)
    y = x ** 2
    art = one_dim_pane.ax.plot(x, y)[0]
    return art


@pytest.fixture(scope='function')
def basic_arts(line, scatter, fit, guide):
    arts = {'line': [{'artist': line,
                      'model_id': 'model_1'}],
            'cursor': [{'artist': scatter,
                        'model_id': 'model_1'}],
            'error_range': list(),
            'crosshair': list(),
            'guide': [{'artist': guide,
                       'model_id': 'model_1'}],
            'patch': list(),
            'fit': [{'artist': fit,
                     'model_id': 'model_1'}]
            }

    return arts


@pytest.fixture(scope='function')
def blank_blitter(qtbot, qapp, mocker):
    mocker.patch.object(QtWidgets, 'QApplication',
                        return_value=qapp)
    fig = mpf.Figure()
    can = mb.FigureCanvasQTAgg(fig)
    art = artists.Artists()
    obj = blitting.BlitManager(can, art)
    return obj


@pytest.fixture(scope='function')
def fig_widget(qtbot, qapp, mocker):
    mocker.patch.object(QtWidgets, 'QApplication',
                        return_value=qapp)
    layout = QtWidgets.QWidget()
    widget = mplwidget.MplWidget(layout)
    return widget


@pytest.fixture(scope='function')
def blank_figure(fig_widget):
    fig = figure.Figure(figure_widget=fig_widget, signals=signals.Signals())
    return fig


@pytest.fixture(scope='function')
def blank_pane():
    p = pane.Pane()
    return p


@pytest.fixture(scope='function')
def blank_onedim():
    p = pane.OneDimPane()
    return p


@pytest.fixture(scope='function')
def blank_view(mocker, qapp, qtbot):
    mocker.patch.object(QtWidgets.QMainWindow, 'show',
                        return_value=None)
    mocker.patch.object(QtWidgets, 'QApplication',
                        return_value=qapp)
    v = view.View(signals.Signals())
    qtbot.add_widget(v)
    return v


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
def line_fit():
    line = models.Linear1D(slope=0.5, intercept=2)
    return line


@pytest.fixture(scope='function')
def gauss_params(gauss_fit):
    params = {'fit': gauss_fit,
              'axis': mpf.Figure().subplots(1, 1),
              'baseline': 10, 'status': 'pass',
              'x_field': 'wavelength', 'x_unit': 'nm',
              'y_field': 'flux', 'y_unit': 'Jy',
              'visible': False, 'lower_limit': 5,
              'upper_limit': 15}
    return {'gauss_file.fits': {1: params}}


@pytest.fixture(scope='function')
def moffat_params(moffat_fit):
    params = {'fit': moffat_fit,
              'axis': mpf.Figure().subplots(1, 1),
              'baseline': 1, 'status': 'pass',
              'x_field': 'wavelength', 'x_unit': 'nm',
              'y_field': 'flux', 'y_unit': 'Jy',
              'visible': False, 'lower_limit': 6,
              'upper_limit': 16}
    return {'moffat_file.fits': {1: params}}


@pytest.fixture(scope='function')
def line_params(line_fit):
    params = {'fit': line_fit,
              'axis': mpf.Figure().subplots(1, 1),
              'status': 'pass',
              'x_field': 'wavelength', 'x_unit': 'nm',
              'y_field': 'flux', 'y_unit': 'Jy',
              'visible': False, 'lower_limit': 6,
              'upper_limit': 16}
    return {'line_file.fits': {1: params}}


@pytest.fixture(scope='function')
def single_gauss_fit():
    gauss = models.Gaussian1D(1, 0, 0.2)
    return gauss


@pytest.fixture(scope='function')
def single_gauss_params(single_gauss_fit):
    params = {'fit': single_gauss_fit,
              'axis': mpf.Figure().subplots(1, 1),
              'baseline': 10, 'status': 'pass',
              'x_field': 'wavelength', 'x_unit': 'nm',
              'y_field': 'flux', 'y_unit': 'Jy',
              'visible': False, 'lower_limit': 5,
              'upper_limit': 15}
    return {'model_id_name': {1: params}}


@pytest.fixture(scope='function')
def gauss_model_fit(gauss_params):
    model = model_fit.ModelFit(gauss_params)
    return model


@pytest.fixture(scope='function')
def moffat_model_fit(moffat_params):
    model = model_fit.ModelFit(moffat_params)
    return model


@pytest.fixture(scope='function')
def line_model_fit(line_params):
    model = model_fit.ModelFit(line_params)
    return model







