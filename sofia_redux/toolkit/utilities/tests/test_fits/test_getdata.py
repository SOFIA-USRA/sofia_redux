# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.toolkit.utilities.fits import getdata


@pytest.fixture
def fits_file(tmpdir):
    filename = str(tmpdir.mkdir('test_getheader').join('test.fits'))
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.arange(10)))
    hdul.append(fits.ImageHDU(data=np.zeros((10, 10))))
    hdul.writeto(filename)
    return filename


def test_badfile(tmpdir):
    assert getdata('__THIS_FILE_DOES_NOT_EXIST__') is None
    badfile = str(tmpdir.mkdir('test_getheader').join('badfits.fits'))

    with open(badfile, 'w') as f:
        f.write('foobar')
    assert getdata(badfile) is None


def test_standard(fits_file):
    d0 = getdata(fits_file, hdu=0)
    d1 = getdata(fits_file, hdu=1)
    assert isinstance(d0, np.ndarray)
    assert isinstance(d1, np.ndarray)
    assert not np.allclose(d0, d1)
