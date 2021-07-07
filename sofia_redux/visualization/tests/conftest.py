# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import astropy.io.fits as pf
import matplotlib
from matplotlib.testing.compare import compare_images
import numpy as np

from sofia_redux.visualization import controller, eye
from sofia_redux.visualization.display import pane

try:
    from PyQt5 import QtWidgets
except ImportError:
    HAS_PYQT5 = False
    QtWidgets = None
else:
    HAS_PYQT5 = True


@pytest.fixture(scope='function')
def log_args():
    args = ['--level', 'DEBUG']
    args = controller.parse_args(args)
    args = controller.check_args(args)
    return args


@pytest.fixture(scope='function')
def simple_axes():
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot()
    return ax


@pytest.fixture(scope='function')
def one_dim_pane_empty(simple_axes):
    return pane.OneDimPane(simple_axes)


@pytest.fixture(scope='function')
def populated_spectral_eye(empty_eye_app, qtbot, mocker, spectral_filenames):
    mocker.patch.object(QtWidgets.QDialog, 'show',
                        return_value=None)
    mocker.patch.object(QtWidgets.QDialog, 'exec_',
                        return_value=None)
    empty_eye_app.add_panes(layout='rows', n_panes=2, kind='spectrum')
    empty_eye_app.load(spectral_filenames)
    empty_eye_app.assign_data('split')
    return empty_eye_app


@pytest.fixture(scope='function')
def populated_multi_order_eye(multi_order_filenames, empty_eye_app):
    empty_eye_app.add_panes(layout='rows', n_panes=2, kind='spectrum')
    empty_eye_app.load(sorted(multi_order_filenames))
    empty_eye_app.assign_data('split')
    return empty_eye_app


@pytest.fixture(scope='function')
def simple_fits_data():
    hdul = pf.HDUList(pf.PrimaryHDU(
        data=np.arange(100).reshape(10, 10)))
    return hdul


@pytest.fixture(scope='function')
def wcs_fits_data():
    header = {'CRPIX1': 1, 'CRVAL1': 0,
              'CRPIX2': 1, 'CRVAL2': 0,
              'CTYPE1': 'RA---TAN',
              'CTYPE2': 'DEC--TAN',
              'CDELT1': 1.0, 'CDELT2': 1.0,
              'EQUINOX': 2000.,
              'BMAJ': 1.0, 'BMIN': 1.0,
              'BPA': 0.0}
    hdul = pf.HDUList(pf.PrimaryHDU(
        data=np.arange(100).reshape(10, 10),
        header=pf.Header(header)))
    return hdul


@pytest.fixture(scope='function')
def cd_wcs_fits_data():
    header = {'CRPIX1': 1, 'CRVAL1': 0,
              'CRPIX2': 1, 'CRVAL2': 0,
              'CTYPE1': 'RA---TAN',
              'CTYPE2': 'DEC--TAN',
              'CD1_1': 1.0, 'CD1_2': 1.0,
              'CD2_1': -1.0, 'CD2_2': 1.0,
              'EQUINOX': 2000.,
              'BMAJ': 1.0, 'BMIN': 1.0,
              'BPA': 0.0}
    hdul = pf.HDUList(pf.PrimaryHDU(
        data=np.arange(100).reshape(10, 10),
        header=pf.Header(header)))
    return hdul


@pytest.fixture(scope='function')
def cube_fits_data():
    header = {'CRPIX1': 1, 'CRVAL1': 0,
              'CRPIX2': 1, 'CRVAL2': 0,
              'CRPIX3': 1, 'CRVAL3': 0,
              'CDELT1': 1.0, 'CDELT2': 1.0, 'CDELT3': 1.0,
              'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec',
              'CUNIT3': 'um'}
    hdul = pf.HDUList(pf.PrimaryHDU(
        data=np.arange(1000).reshape(10, 10, 10),
        header=pf.Header(header)))
    return hdul


@pytest.fixture(scope='function')
def spectrum_fits_data():
    header = {'XUNITS': 'um', 'YUNITS': 'Jy'}
    hdul = pf.HDUList(pf.PrimaryHDU(
        data=np.arange(500).reshape(5, 100),
        header=pf.Header(header)))
    return hdul


@pytest.fixture(scope='function')
def figures_same(tmpdir):
    def test(fig1, fig2):
        t1 = str(tmpdir.join('test1.png'))
        t2 = str(tmpdir.join('test2.png'))
        fig1.savefig(t1)
        fig2.savefig(t2)
        return compare_images(t1, t2, 0) is None
    return test
