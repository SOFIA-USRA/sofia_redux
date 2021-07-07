# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.toolkit.image.adjust import rotate90
from sofia_redux.spectroscopy.readflat import readflat


@pytest.fixture
def data():
    header = fits.header.Header()
    expected = {}
    header['PLTSCALE'] = 1.1
    header['SLTH_ARC'] = 1.2
    header['SLTH_PIX'] = 1.3
    header['SLTW_ARC'] = 1.4
    header['SLTW_PIX'] = 1.5
    header['RP'] = 999
    header['ROTATION'] = 2
    header['EDGEDEG'] = 1
    header['MODENAME'] = 'testmode'
    header['NORDERS'] = 2
    expected.update(header)
    expected['PS'] = expected.pop('PLTSCALE')
    expected['SLITH_ARC'] = expected.pop('SLTH_ARC')
    expected['SLITH_PIX'] = expected.pop('SLTH_PIX')
    expected['SLITW_ARC'] = expected.pop('SLTW_ARC')
    expected['SLITW_PIX'] = expected.pop('SLTW_PIX')
    expected['PS'] = header['PLTSCALE']
    header['ORDERS'], expected['ORDERS'] = '1,2', [1, 2]
    header['OR001_XR'] = '1,8'
    header['OR002_XR'] = '2,7'
    expected['XRANGES'] = [[1, 8], [2, 7]]
    header['OR001_B1'] = 1.0
    header['OR001_B2'] = 0.0
    header['OR001_T1'] = 8.0
    header['OR001_T2'] = 0.0
    header['OR002_B1'] = 2.0
    header['OR002_B2'] = 0.0
    header['OR002_T1'] = 7.0
    header['OR002_T2'] = 0.0
    expected['EDGECOEFFS'] = [[[1.0, 0], [8.0, 0]],
                              [[2.0, 0], [7.0, 0]]]
    header['OR001RMS'] = 1000.0
    header['OR002RMS'] = 2000.0
    expected['rms'] = [1000.0, 2000.0]
    image, _ = np.mgrid[:10, :10]
    image = image.astype(float)
    var, flags = image.copy() + 100, image.copy() + 200
    expected['image'] = rotate90(image, header['ROTATION'])
    expected['variance'] = rotate90(var, header['ROTATION'])
    expected['flags'] = rotate90(flags, header['ROTATION'])
    omask = np.full((10, 10), 0)
    omask[1:9, 1:9] = 1
    omask[2:8, 2:8] = 2
    expected['omask'] = omask
    return image, var, flags, header, expected


@pytest.fixture
def ishell(data, tmpdir):
    image, var, flags, header, expected = data
    hdul = fits.HDUList(
        [fits.PrimaryHDU(header=header), fits.ImageHDU(image),
         fits.ImageHDU(var), fits.ImageHDU(flags)])
    filename = str(tmpdir.join('ishell_test.fits'))
    hdul.writeto(filename)
    return filename, expected


@pytest.fixture
def pre_ishell(data, tmpdir):
    image, var, flags, header, expected = data
    newheader = fits.header.Header()
    for k, v in header.items():
        if k.startswith('OR0'):
            newheader['ODR' + k[3:]] = v
        else:
            newheader[k] = v
    data = np.array([image, var, flags])
    hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=newheader)])
    filename = str(tmpdir.join('pre_ishell_test.fits'))
    hdul.writeto(filename)
    return filename, expected


@pytest.fixture
def defaults(data, tmpdir):
    image, var, flags, header, expected = data
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(image),
         fits.ImageHDU(var), fits.ImageHDU(flags)])
    filename = str(tmpdir.join('default.fits'))
    for k, v in expected.items():
        if isinstance(v, np.ndarray):
            expected[k] *= 0
        else:
            expected[k] = 0
    expected['image'] = image
    expected['variance'] = var
    expected['flags'] = flags
    expected['rms'] = np.zeros(0)
    expected['XRANGES'] = np.zeros(0)
    expected['EDGECOEFFS'] = np.zeros(0)
    expected['MODENAME'] = 'None'
    hdul.writeto(filename)
    return filename, expected


def test_failure():
    assert readflat('__not_a_file__') is None


def test_ishell(ishell):
    filename, expected = ishell
    result = readflat(filename)
    for k, v in expected.items():
        if isinstance(v, (list, np.ndarray)):
            assert np.allclose(result[k.lower()], v)
        else:
            assert result[k.lower()] == v


def test_pre_ishell(pre_ishell):
    filename, expected = pre_ishell
    result = readflat(filename)
    for k, v in expected.items():
        if isinstance(v, (list, np.ndarray)):
            assert np.allclose(result[k.lower()], v)
        else:
            assert result[k.lower()] == v


def test_defaults(defaults):
    filename, expected = defaults
    result = readflat(filename)
    for k, v in expected.items():
        if isinstance(v, (list, np.ndarray)):
            if len(v) != 0:
                assert np.allclose(result[k.lower()], v)
            else:
                assert len(result[k.lower()]) == 0
        else:
            assert result[k.lower()] == v
