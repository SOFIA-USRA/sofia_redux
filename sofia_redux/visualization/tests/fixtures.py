#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy

import pytest
import numpy as np
import astropy.io.fits as pf
from PyQt5 import QtWidgets, QtCore
from astropy.modeling import models
from matplotlib import figure as mpf

from sofia_redux.visualization import eye, signals
from sofia_redux.visualization.display import view
from sofia_redux.visualization.tests import resources

__all__ = ['atran_params', 'atran_name', 'atran_hdul', 'atran_file',
           'grism_name', 'grism_hdul', 'multi_ap_grism_hdul',
           'multiorder_name_merged', 'multiorder_name_spec',
           'multiorder_hdul_merged', 'multiorder_hdul_spec',
           'forcast_image_name', 'forcast_hdul_image',
           'forcast_fits_image', 'exes_fits_image', 'multi_order_filenames',
           'spectral_filenames', 'spectral_hduls', 'split_order_hdul',
           'combined_order_hdul', 'spectrum_hdu', 'image_hdu', 'loaded_eye',
           'loaded_eye_with_alt', 'empty_eye_app', 'empty_view', 'open_mock']


@pytest.fixture(scope='function')
def atran_params():
    params = {'altitude': 40000, 'za': 55, 'pwv': 40,
              'min_wave': 40, 'max_wave': 300}
    return params


@pytest.fixture(scope='function')
def atran_name(atran_params):
    name = pf.file._File()
    alt = f"{atran_params['altitude'] // 1000:d}K"
    za = f"{atran_params['za']:d}deg"
    pwv = f"{atran_params['pwv']:d}pwv"
    wave = f"{atran_params['min_wave']:d}-{atran_params['max_wave']:d}mum"
    name.name = f'atran_{alt}_{za}_{pwv}_{wave}.fits'
    return name


@pytest.fixture(scope='function')
def atran_hdul(atran_params, atran_name):
    hdul = resources.atran_data(atran_params)
    hdul._file = atran_name
    return hdul


@pytest.fixture(scope='function')
def atran_file(atran_hdul, atran_name, tmp_path):
    filename = str(tmp_path / atran_name.name)
    atran_hdul.writeto(filename)
    return filename


@pytest.fixture(scope='function')
def grism_name():
    name = pf.file._File()
    name.name = 'grism_COA_100.fits'
    return name


@pytest.fixture(scope='function')
def grism_hdul(grism_name):
    hdul = resources.forcast_data()
    header = hdul[0].header

    header['FILENAME'] = grism_name.name
    header['INSTCFG'] = 'GRISM_SWC'
    header['INSTMODE'] = 'C2N'
    header['SKYMODE'] = 'NMC'
    header['SPECTEL1'] = 'FOR_G063'
    header['SPECTEL2'] = 'FOR_F315'
    header['DETCHAN'] = 0
    header['SLIT'] = 'FOR_L524'
    header['PRODTYPE'] = 'coadded_spectrum'
    header['XUNIT'] = 'um'
    header['YUNIT'] = 'Jy'

    extensions = ['FLUX', 'ERROR', 'EXPOSURE', 'WAVEPOS',
                  'SPECTRAL_FLUX', 'SPECTRAL_ERROR',
                  'TRANSMISSION', 'RESPONSE']
    for i in range(len(hdul), len(extensions)):
        hdu = hdul[0].copy()
        hdul.append(hdu)
    for hdu, extension in zip(hdul, extensions):
        hdu.name = extension
        hdu.header['EXTNAME'] = extension

    shape = (296, 240)
    hdul[0].data = _grism_flux(shape)
    hdul[1].data = _grism_flux_error(shape)
    hdul[2].data = _grism_exposure(shape)
    hdul[3].data = _grism_wavepos(shape)
    hdul[4].data = _grism_spectral_flux(shape)
    hdul[5].data = _grism_spectral_error(shape)
    hdul[6].data = _grism_transmission(shape)
    hdul[7].data = _grism_response(shape)

    hdul._file = grism_name

    return hdul


@pytest.fixture(scope='function')
def multi_ap_grism_hdul(grism_hdul):
    ap_ext = [4, 5, 6, 7]
    for ext in ap_ext:
        one_spec = grism_hdul[ext].data
        two_spec = grism_hdul[ext].data.copy()
        grism_hdul[ext].data = np.array([one_spec, two_spec])
    return grism_hdul


def _grism_flux(shape):
    data = np.zeros(shape)
    noise_std = 0.5
    peak_locations = [45, 90, 135, 180, 225, 270]
    peak_heights = [-2, -1, 3, 3, -1, -1]
    sigma = 2
    full_model = models.Const1D(amplitude=0)
    for loc, height in zip(peak_locations, peak_heights):
        full_model = full_model + models.Gaussian1D(amplitude=height,
                                                    mean=loc, stddev=sigma)
    col_indices = np.arange(data.shape[1])
    rows = np.arange(data.shape[0])
    col_shape = full_model(rows)
    for col in col_indices:
        data[:, col] = col_shape + np.random.normal(0, noise_std, len(rows))
    return data


def _grism_flux_error(shape):
    data = np.random.normal(0, 0.5, shape)
    return data


def _grism_exposure(shape):
    data = np.ones(shape) * 30
    return data


def _grism_wavepos(shape):
    min_wavepos = 4.95
    max_wavepos = 7.85
    data = np.linspace(min_wavepos, max_wavepos, shape[0])
    return data


def _grism_spectral_flux(shape):
    feature = models.Gaussian1D(amplitude=5, mean=shape[0] // 2, stddev=20)
    base = models.Linear1D(slope=0.05, intercept=30)
    combined = feature + base
    x = np.arange(shape[0])
    data = combined(x) + np.random.normal(0, 1, shape[0])
    return data


def _grism_spectral_error(shape):
    data = np.abs(np.random.normal(0, 1, shape[0]))
    return data


def _grism_transmission(shape):
    x = np.arange(shape[0])
    feature = models.Gaussian1D(amplitude=0.3, mean=shape[0] // 3,
                                stddev=20)
    feature = feature(x) + np.random.normal(0, 0.05, shape[0])
    data = 1 - np.abs(feature)
    return data


def _grism_response(shape):
    x = np.arange(shape[0])
    base = models.Const1D(0.004)
    data = base(x) + np.random.normal(0, 1e-4, shape[0])
    return data


@pytest.fixture(scope='function')
def multiorder_name_merged():
    name = pf.file._File()
    name.name = 'grism_MRD_100.fits'
    return name


@pytest.fixture(scope='function')
def multiorder_name_spec():
    name = pf.file._File()
    name.name = 'grism_SPC_100.fits'
    return name


@pytest.fixture(scope='function')
def multiorder_hdul_merged(multiorder_name_merged):
    hdul = resources.exes_data()
    hdul.pop(1)
    header = hdul[0].header

    header['FILENAME'] = multiorder_name_merged.name
    header['SKYMODE'] = 'None'
    header['DETCHAN'] = None
    header['SLIT'] = 'EXE_S32'
    header['PRODTYPE'] = 'mrgordspec'
    header['NORDERS'] = 1

    hdul[0].data = _multiorder_merged_data()
    hdul._file = multiorder_name_merged

    return hdul


@pytest.fixture(scope='function')
def multiorder_hdul_spec(multiorder_name_spec):
    hdul = resources.exes_data()
    hdul.pop(1)
    header = hdul[0].header

    header['FILENAME'] = multiorder_name_spec.name
    header['SKYMODE'] = 'None'
    header['DETCHAN'] = None
    header['SLIT'] = 'EXE_S32'
    header['PRODTYPE'] = 'spec'
    header['NORDERS'] = 10
    header['XUNITS'] = 'cm-1'
    header['YUNITS'] = 'erg s-1 cm-2 sr-1 (cm-1)-1'

    hdul[0].data = _multiorder_spec_data(header['NORDERS'])
    hdul._file = multiorder_name_spec

    return hdul


def _multiorder_merged_data():
    size = 1000
    min_wave = 1300
    max_wave = 1335

    wavelength = np.linspace(min_wave, max_wave, size)

    n_peaks = 10
    peak_locations = np.random.uniform(min_wave, max_wave, n_peaks)
    peak_heights = np.random.uniform(0, 1.5, n_peaks)
    sigma = 2
    flux_model = models.Const1D(amplitude=2)
    error_model = models.Const1D(amplitude=0)
    transmission_model = models.Const1D(amplitude=1)

    for loc, height in zip(peak_locations, peak_heights):
        flux_model = flux_model - models.Gaussian1D(amplitude=height,
                                                    mean=loc, stddev=sigma)
        error_model = error_model + models.Gaussian1D(mean=loc,
                                                      amplitude=height * 0.1,
                                                      stddev=sigma)
        transmission_model = transmission_model - models.Gaussian1D(
            mean=loc, amplitude=height * 0.5, stddev=sigma)

    flux = flux_model(wavelength)
    error = error_model(wavelength)
    transmission = transmission_model(wavelength)
    data = np.vstack([wavelength, flux, error, transmission]).T
    return data


def _multiorder_spec_data(norders):
    size = 1000
    min_wave = 1300
    max_wave = 1335
    subrange_size = (max_wave - min_wave) / norders
    subrange_npoints = size // norders
    data = np.zeros((norders, 4, subrange_npoints))
    for order_num in range(norders):
        start_wave = min_wave + order_num * subrange_size
        end_wave = start_wave + subrange_size
        waves = np.linspace(start_wave, end_wave, subrange_npoints)
        peak_loc = np.median(waves)
        height = np.random.uniform(0, 1.5)
        sigma = np.random.uniform(0, .75)

        flux_model = models.Const1D(amplitude=2)
        error_model = models.Const1D(amplitude=0)
        transmission_model = models.Const1D(amplitude=1)

        flux_model = flux_model - models.Gaussian1D(amplitude=height,
                                                    mean=peak_loc,
                                                    stddev=sigma)
        error_model = error_model + models.Gaussian1D(mean=peak_loc,
                                                      amplitude=height * 0.1,
                                                      stddev=sigma)
        transmission_model = transmission_model - models.Gaussian1D(
            mean=peak_loc, amplitude=height * 0.5, stddev=sigma)

        data[order_num, 0, :] = waves
        data[order_num, 1, :] = flux_model(waves)
        data[order_num, 2, :] = error_model(waves)
        data[order_num, 3, :] = transmission_model(waves)

    return data


@pytest.fixture(scope='function')
def forcast_image_name():
    name = pf.file._File()
    name.name = 'image_COA_100.fits'
    return name


@pytest.fixture(scope='function')
def forcast_hdul_image(forcast_image_name):
    hdul = resources.forcast_data()
    hdul[0].name = 'FLUX'
    header = hdul[0].header
    header['FILENAME'] = forcast_image_name.name
    hdul._file = forcast_image_name
    return hdul


@pytest.fixture(scope='function')
def forcast_fits_image(forcast_hdul_image, forcast_image_name, tmp_path):
    filename = str(tmp_path / forcast_image_name.name)
    forcast_hdul_image.writeto(filename)
    return filename


@pytest.fixture(scope='function')
def exes_fits_image(multiorder_hdul_spec, multiorder_name_spec, tmp_path):
    filename = str(tmp_path / multiorder_name_spec.name)
    multiorder_hdul_spec.writeto(filename)
    return filename


@pytest.fixture(scope='function')
def multi_order_filenames(multiorder_hdul_spec, tmp_path):
    n_exes = 3
    filenames = list()
    for i in range(n_exes):
        name = pf.file._File()
        name.name = f'grism_SPC_{i + 1}00.fits'
        filename = str(tmp_path / name.name)
        hdul = copy.copy(multiorder_hdul_spec)
        hdul[0].header['FILENAME'] = name.name
        hdul._file = name
        hdul.writeto(filename)
        hdul.close()
    return filenames


@pytest.fixture(scope='function')
def spectral_filenames(grism_hdul, multiorder_hdul_spec, tmp_path):
    n_forcast = 3
    n_exes = 2
    filename = list()
    for i in range(n_forcast):
        name = pf.file._File()
        name.name = f'grism_COA_{i + 1}00.fits'
        hdul = copy.copy(grism_hdul)
        hdul._file = name
        hdul[0].header['FILENAME'] = name.name
        path = str(tmp_path / name.name)
        hdul.writeto(path)
        hdul.close()
        filename.append(path)

    for i in range(n_exes):
        name = pf.file._File()
        name.name = f'grism_SPC_{i + 1}00.fits'
        hdul = copy.copy(multiorder_hdul_spec)
        hdul._file = name
        hdul[0].header['FILENAME'] = name.name
        path = str(tmp_path / name.name)
        hdul.writeto(path)
        hdul.close()
        filename.append(path)

    grism_hdul.close()
    multiorder_hdul_spec.close()
    return filename


@pytest.fixture(scope='function')
def spectral_hduls(spectral_filenames):
    hduls = [pf.open(fn, memmap=False) for fn in spectral_filenames]
    return hduls


@pytest.fixture(scope='function')
def split_order_hdul():
    hdul = pf.HDUList()
    filename = pf.file._File()
    filename.name = 'split_order.fits'
    hdul._file = filename

    extensions = ['wavepos', 'spectral_flux', 'spectral_error']
    pixels = 100
    waves = np.linspace(5, 20, pixels)
    flux_model = models.Const1D(2) + models.Gaussian1D(1, 10, 2)

    error = np.abs(np.random.normal(0, 0.1, pixels))

    datasets = [waves, flux_model(waves), error]
    for dataset, extension in zip(datasets, extensions):
        hdu = pf.ImageHDU()
        hdu.name = extension.upper()
        hdu.data = dataset
        hdul.append(hdu)
    return hdul


@pytest.fixture(scope='function')
def combined_order_hdul(split_order_hdul):
    data = list()
    for hdu in split_order_hdul:
        data.append(hdu.data)

    hdul = pf.HDUList()
    hdu = pf.ImageHDU()
    filename = pf.file._File()
    filename.name = 'combined_order.fits'
    hdul._file = filename
    hdu.data = np.vstack(data)
    header = split_order_hdul[0].header
    header['XUNITS'] = 'um'
    header['YUNITS'] = 'Jy'
    hdu.header = header.copy()

    hdul.append(hdu)

    return hdul


@pytest.fixture(scope='function')
def spectrum_hdu():
    hdu = pf.ImageHDU()
    hdu.name = 'SPECTRAL_FLUX'
    pixels = 100
    waves = np.linspace(5, 20, pixels)
    flux_model = models.Const1D(2) + models.Gaussian1D(1, 10, 2)
    hdu.data = flux_model(waves)
    hdu.header['YUNIT'] = 'Jansky'
    return hdu


@pytest.fixture(scope='function')
def image_hdu():
    hdu = pf.ImageHDU()
    hdu.name = 'FLUX'
    shape = (100, 100)
    npoints = 1000
    model = models.Gaussian2D(amplitude=2, x_mean=shape[0] // 2,
                              y_mean=shape[1] // 2,
                              x_stddev=3, y_stddev=3)
    x = np.linspace(0, shape[0], npoints)
    y = np.linspace(0, shape[1], npoints)
    xx, yy = np.meshgrid(x, y)
    flux = model(xx, yy)
    noise = np.random.normal(0, 0.1, size=flux.size).reshape(flux.shape)

    hdu.data = flux + noise
    hdu.header['BUNIT'] = 'Jy'
    return hdu


@pytest.fixture(scope='function')
def empty_view(mocker, qtbot, qapp):
    mocker.patch.object(QtWidgets.QMainWindow, 'show',
                        return_value=None)
    mocker.patch.object(QtWidgets, 'QApplication',
                        return_value=qapp)
    mocker.patch.object(QtWidgets.QApplication, 'exec_',
                        return_value=None)
    view_ = view.View(signals.Signals())
    qtbot.add_widget(view_)
    return view_


@pytest.fixture(scope='function')
def empty_eye_app(qtbot, qapp, empty_view, log_args):
    eye_app = eye.Eye(log_args, view_=empty_view)
    # set terminal print level to debug
    return eye_app


@pytest.fixture(scope='function')
def loaded_eye(qapp, mocker, qtbot, spectral_filenames):
    mocker.patch.object(QtWidgets.QMainWindow, 'show',
                        return_value=None)
    mocker.patch.object(QtWidgets.QMainWindow, 'raise_',
                        return_value=None)
    mocker.patch.object(QtWidgets, 'QApplication',
                        return_value=qapp)
    mocker.patch.object(QtWidgets.QFileDialog,
                        'getOpenFileNames',
                        return_value=[spectral_filenames])

    app = eye.Eye()
    qtbot.mouseClick(app.view.add_file_button, QtCore.Qt.LeftButton)
    app.view.refresh_controls()
    app.view.file_table_widget.selectRow(0)
    mocker.patch.object(app.view.file_table_widget, 'hasFocus',
                        return_value=True)
    qtbot.keyClick(app.view.file_table_widget,
                   QtCore.Qt.Key_Return)
    app.view.refresh_controls()
    app.view.open_eye()
    qtbot.wait(1000)
    return app


@pytest.fixture(scope='function')
def loaded_eye_with_alt(qapp, mocker, qtbot, spectral_filenames):
    mocker.patch.object(QtWidgets.QMainWindow, 'show',
                        return_value=None)
    mocker.patch.object(QtWidgets.QMainWindow, 'raise_',
                        return_value=None)
    mocker.patch.object(QtWidgets, 'QApplication',
                        return_value=qapp)
    mocker.patch.object(QtWidgets.QFileDialog,
                        'getOpenFileNames',
                        return_value=[spectral_filenames])

    app = eye.Eye()
    qtbot.mouseClick(app.view.add_file_button, QtCore.Qt.LeftButton)
    app.view.refresh_controls()
    app.view.file_table_widget.selectRow(0)
    mocker.patch.object(app.view.file_table_widget, 'hasFocus',
                        return_value=True)
    qtbot.keyClick(app.view.file_table_widget,
                   QtCore.Qt.Key_Return)
    qtbot.mouseClick(app.view.enable_overplot_checkbox,
                     QtCore.Qt.LeftButton)
    app.view.axes_selector.setCurrentText('Current Overplot')
    app.view.y_property_selector.setCurrentText('spectral_flux')
    app.view.refresh_controls()
    app.view.open_eye()
    qtbot.wait(1000)
    return app


@pytest.fixture(scope='function')
def open_mock(mocker, qapp):
    mocker.patch.object(QtWidgets.QApplication, 'exec_',
                        return_value=0)
    mocker.patch.object(QtWidgets, 'QApplication',
                        return_value=qapp)
    open_mock = mocker.patch.object(eye.Eye, 'open_eye')
    return open_mock


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
def gauss_params(gauss_fit):
    params = {'fit': gauss_fit,
              'axis': mpf.Figure().subplots(1, 1),
              'baseline': 10, 'status': 'pass',
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
              'baseline': 10, 'status': 'pass',
              'x_field': 'wavelength', 'x_unit': 'nm',
              'y_field': 'flux', 'y_unit': 'Jy',
              'visible': False, 'lower_limit': 5,
              'upper_limit': 15}
    return {'model_id_name': {1: params}}
