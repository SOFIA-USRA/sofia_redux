# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.spectroscopy.wave_calibration \
    import (readwavecalinfo, BaseWavecal, Wavecal, Wavecal1D,
            Wavecal1DXD, Wavecal2D, Wavecal2DXD)
import sofia_redux.spectroscopy.tests
from sofia_redux.toolkit.interpolate import tabinv


@pytest.fixture
def wcal_info_file():
    return os.path.join(
        os.path.dirname(sofia_redux.spectroscopy.tests.__file__),
        'data', 'H1_wavecalinfo.fits')


@pytest.fixture
def lines_file():
    return os.path.join(
        os.path.dirname(sofia_redux.spectroscopy.tests.__file__),
        'data', 'H1_lines.dat')


def test_load_types(wcal_info_file):
    # read default file
    wcal = readwavecalinfo(wcal_info_file)

    # check type returned
    assert isinstance(wcal, BaseWavecal)
    assert isinstance(wcal, Wavecal)
    assert isinstance(wcal, Wavecal2DXD)
    assert not hasattr(wcal, 'rms')

    # modify to check other types
    hdul = fits.open(wcal_info_file)
    hdul[0].header['WCALTYPE'] = '1D'
    hdul[0].header['FITRMS'] = 1.0
    wcal = readwavecalinfo(hdul)
    assert isinstance(wcal, Wavecal1D)
    assert np.allclose(wcal.rms, 1)

    hdul[0].header['WCALTYPE'] = '1DXD'
    wcal = readwavecalinfo(hdul)
    assert isinstance(wcal, Wavecal1DXD)
    assert not hasattr(wcal, 'rms')

    hdul[0].header['WCALTYPE'] = '2D'
    wcal = readwavecalinfo(hdul)
    assert isinstance(wcal, Wavecal2D)
    assert np.allclose(wcal.rms, 1)

    # invalid type
    hdul[0].header['WCALTYPE'] = 'BAD'
    with pytest.raises(ValueError) as err:
        readwavecalinfo(hdul)
    assert 'Invalid wave calibration type' in str(err)


def test_linedeg(wcal_info_file):
    # read default file
    wcal = readwavecalinfo(wcal_info_file)
    assert not hasattr(wcal, 'c2xdeg')
    assert not hasattr(wcal, 'c2ydeg')

    # modify linedeg for 2DXD
    hdul = fits.open(wcal_info_file)
    hdul[0].header['LINEDEG'] = 2
    hdul[0].header['C2XDEG'] = 3
    hdul[0].header['C2YDEG'] = 4
    wcal = readwavecalinfo(hdul)
    assert wcal.c2xdeg == 3
    assert wcal.c2ydeg == 4

    # and for 2D
    hdul[0].header['WCALTYPE'] = '2D'
    hdul[0].header['FITRMS'] = 1.0
    wcal = readwavecalinfo(hdul)
    assert wcal.c2xdeg == 3
    assert not hasattr(wcal, 'c2ydeg')


def test_load_errors(wcal_info_file):
    # missing file
    with pytest.raises(ValueError) as err:
        readwavecalinfo('badfile.fits')
    assert 'Could not load' in str(err)

    # missing data
    hdul = fits.HDUList(fits.PrimaryHDU(header=fits.Header({'test': 1})))
    with pytest.raises(ValueError) as err:
        readwavecalinfo(hdul)
    assert 'No data present' in str(err)

    # bad xcor order
    hdul = fits.open(wcal_info_file)
    hdul[0].header['XCORORDR'] = -2
    with pytest.raises(ValueError) as err:
        readwavecalinfo(hdul)
    assert 'X-correlation order -2 not found' in str(err)


def test_guess_lines(wcal_info_file, lines_file):
    wcal = readwavecalinfo(wcal_info_file)
    assert wcal.lines.table is None

    # can't set offset if table is none
    wcal.lines.offset = 270
    assert np.allclose(wcal.lines.offset, 0)

    # try to guess lines
    with pytest.raises(ValueError) as err:
        wcal.guess_lines()
    assert 'Line list not loaded' in str(err)

    # load line list
    wcal.load_lines(lines_file)
    assert wcal.lines.table is not None
    assert 'wavelength' in wcal.lines.table
    assert 'xguess' not in wcal.lines.table
    assert len(wcal.lines.table) == 27

    # guess lines
    wcal.guess_lines()
    assert 'xguess' in wcal.lines.table
    assert 'xwindow' in wcal.lines.table

    # test guesses against expected:
    # should be a single shift value for all,
    # corresponding to the starting xrange
    def check_diff(value):
        getdata = zip(
            wcal.lines.table['order'],
            wcal.lines.table['wavelength'],
            wcal.lines.table['xguess'])
        diff = []
        for od, wv, xg in getdata:
            idx = list(wcal.orders).index(od)
            check = tabinv(wcal.data[idx][0], wv)
            diff.append(xg - check)
        assert np.allclose(np.mean(diff), value)
        assert np.allclose(np.std(diff), 0)

    # expected shift is 270.
    check_diff(270.)

    # offset is currently zero
    assert np.allclose(wcal.lines.offset, 0.0)

    # set it to -270: is applied to the lines in the table
    wcal.lines.offset = -270
    check_diff(0.)


def test_line_error(wcal_info_file, lines_file):
    wcal = readwavecalinfo(wcal_info_file)
    lineinfo = wcal.lines
    orders = wcal.orders
    xranges = wcal.xranges
    wavelengths = wcal.data[:, 0]

    # try to read a missing file
    with pytest.raises(ValueError) as err:
        lineinfo.readlinelist('badfile.dat')
    assert 'Unable to load' in str(err)

    # try to xguess without a table
    with pytest.raises(ValueError) as err:
        lineinfo.xguess(wavelengths, xranges, orders)
    assert 'Line list not loaded' in str(err)

    # read a good one
    lineinfo.readlinelist(lines_file)

    # xguess with mismatched input
    with pytest.raises(ValueError) as err:
        lineinfo.xguess(wavelengths, xranges, orders[1:])
    assert 'Orders and wavelength axis 0 size mismatch' in str(err)
    with pytest.raises(ValueError) as err:
        lineinfo.xguess(wavelengths, xranges[1:], orders)
    assert 'Orders and xranges axis 0 size mismatch' in str(err)
