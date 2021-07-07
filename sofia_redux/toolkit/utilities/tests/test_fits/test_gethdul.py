# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.toolkit.utilities.fits import gethdul


@pytest.fixture
def fits_file(tmpdir):
    filename = str(tmpdir.mkdir('test_robust_read').join('test.fits'))
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.arange(10)))
    hdul.append(fits.ImageHDU(data=np.zeros((10, 10))))
    hdul.writeto(filename)
    return filename


def test_keep(fits_file):
    hdul = fits.open(fits_file)
    assert gethdul(hdul) is hdul


def test_list(fits_file):
    hdul = fits.open(fits_file)
    hdul_list = [hdul[0], hdul[1]]
    hdul2 = gethdul(hdul_list)
    assert isinstance(hdul2, fits.HDUList)
    assert np.allclose(hdul2[0].data, hdul[0].data)
    assert np.allclose(hdul2[1].data, hdul[1].data)


def test_standard(fits_file):
    result = gethdul(fits_file)
    assert isinstance(result, fits.HDUList)
    assert np.allclose(result[0].data, np.arange(10))
    assert result[1].data.shape == (10, 10)


def test_error(fits_file):
    assert gethdul('__does_not_exist__') is None
    with open(fits_file, 'r') as f:
        contents = f.read()

    idx_key = contents.find('NAXIS1')

    with open(fits_file, 'r+b') as f:
        f.seek(idx_key)
        f.write(b'\x00')

    assert gethdul(fits_file) is None
