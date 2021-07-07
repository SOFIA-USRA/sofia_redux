# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.spectroscopy.readwavecal import readwavecal


@pytest.fixture
def testfile(tmpdir):
    y, x = np.mgrid[:8, :8]
    y = y + 0.0
    x = x + 100.0
    data = np.array([x, y])
    filename = str(tmpdir.join('testfile.fits'))
    header = fits.header.Header()
    header['ORDERS'] = '1,2'
    header['NORDERS'] = 2
    header['WCTYPE'] = '2D'
    header['WAVEFMT'] = 'testwave'
    header['SPATFMT'] = 'testspat'
    header['DISP01'] = 1.1
    header['DISP02'] = 1.2
    header['WXDEG'] = 1
    header['WYDEG'] = 2
    header['SXDEG'] = 0
    header['SYDEG'] = 1
    header['WDEG'] = 10
    header['ODEG'] = 11
    header['HOMEORDR'] = 12
    header['1DWC01'] = 101
    header['1DWC02'] = 102
    header['OR01WC01'] = 0.1
    header['OR01WC02'] = 0.2
    header['OR01WC03'] = 0.3
    header['OR01WC04'] = 0.4
    header['OR01WC05'] = 0.5
    header['OR01WC06'] = 0.6
    header['OR02WC01'] = 1.1
    header['OR02WC02'] = 1.2
    header['OR02WC03'] = 1.3
    header['OR02WC04'] = 1.4
    header['OR02WC05'] = 1.5
    header['OR02WC06'] = 1.6
    header['OR01SC01'] = 2.1
    header['OR01SC02'] = 2.2
    header['OR02SC01'] = 3.1
    header['OR02SC02'] = 3.2
    hdul = fits.HDUList(fits.PrimaryHDU(data))
    hdul[0].header.update(header)
    hdul.writeto(filename)
    return filename


def test_failure():
    assert readwavecal('__not_a_file__') is None


def test_success(testfile):
    filename = testfile
    info = {}
    wavecal, spatcal = readwavecal(filename, rotate=2, info=info)
    assert np.allclose(wavecal[0],
                       [107, 106, 105, 104, 103, 102, 101, 100])
    assert np.allclose(spatcal[:, 0],
                       [7, 6, 5, 4, 3, 2, 1, 0])
    assert info['wctype'] == '2D'
    assert np.allclose(info['orders'], [1, 2])
    assert info['norders'] == 2
    assert info['wdeg'] == 10
    assert info['odeg'] == 11
    assert info['homeordr'] == 12
    assert info['wxdeg'] == 1
    assert info['wydeg'] == 2
    assert info['sxdeg'] == 0
    assert info['sydeg'] == 1
    assert np.allclose(info['xo2w'], [101, 102])
    assert np.allclose(info['xy2w'],
                       [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]])
    assert np.allclose(info['xy2s'],
                       [[2.1, 2.2], [3.1, 3.2]])


def test_zeros(testfile):
    filename = testfile
    hdul = fits.open(filename)
    hdul[0].header['WDEG'] = 0
    hdul[0].header['ODEG'] = 0
    hdul[0].header['WXDEG'] = 0
    hdul[0].header['WYDEG'] = 0
    hdul[0].header['SXDEG'] = 0
    hdul[0].header['SYDEG'] = 0

    info = {}
    readwavecal(hdul, rotate=2, info=info)
    assert np.allclose(info['xo2w'], 0)
    assert np.allclose(info['xy2w'], 0)
    assert np.allclose(info['xy2s'], 0)
    assert info['xo2w'].shape == (0,)
    assert info['xy2w'].shape == (2, 0)
    assert info['xy2s'].shape == (2, 0)
