# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.toolkit.utilities.fits import robust_read


@pytest.fixture
def fits_file(tmpdir):
    filename = str(tmpdir.mkdir('test_robust_read').join('test.fits'))
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.arange(10)))
    hdul.append(fits.ImageHDU(data=np.zeros((10, 10))))
    hdul[0].header['TESTKEY'] = '__a_test_value__'
    hdul.writeto(filename)
    return filename


def test_badfile():
    result = robust_read('__this_file_does_not_exist__', verbose=True)
    assert result == (None, None)


def test_missing_hdu(fits_file):
    result = robust_read(fits_file, data_hdu=100)
    assert result == (None, None)
    result = robust_read(fits_file, header_hdu=100)
    assert result == (None, None)


def test_standard_read(fits_file):
    data0, header0 = robust_read(fits_file, verbose=True)
    assert np.allclose(data0, np.arange(10))
    assert isinstance(header0, fits.Header)
    data1, header1 = robust_read(fits_file, extension=1)
    assert np.allclose(data1, 0)
    assert data1.shape == (10, 10)
    assert isinstance(header1, fits.Header)
    assert not fits.HeaderDiff(header0, header1).identical
    assert not np.allclose(data0, data1)

    data11, header00 = robust_read(fits_file, header_hdu=0, data_hdu=1)
    assert np.allclose(data11, data1)
    assert fits.HeaderDiff(header0, header00).identical


def test_corrupt_fits_file_fix(fits_file):
    with open(fits_file, 'r') as f:
        contents = f.read()

    idx_value = contents.find('__a_test_value__')
    idx_key = contents.find('TESTKEY')

    with open(fits_file, 'r+b') as f:
        f.seek(idx_value)
        f.write(b'\x00')

    data, header = robust_read(fits_file)
    assert header['TESTKEY'] == 'UNKNOWN'

    with open(fits_file, 'r+b') as f:
        f.seek(idx_key)
        f.write(b'\x00')

    data, header = robust_read(fits_file)
    assert data is None
    assert header is None
