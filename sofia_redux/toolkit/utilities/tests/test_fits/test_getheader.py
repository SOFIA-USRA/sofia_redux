# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.toolkit.utilities.fits import getheader


@pytest.fixture
def fits_file(tmpdir):
    filename = str(tmpdir.mkdir('test_getheader').join('test.fits'))
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.arange(10)))
    hdul.append(fits.ImageHDU(data=np.zeros((10, 10))))
    hdul.writeto(filename)
    return filename


def test_badfile(tmpdir):
    assert getheader('__THIS_FILE_DOES_NOT_EXIST__') is None
    badfile = str(tmpdir.mkdir('test_getheader').join('badfits.fits'))

    with open(badfile, 'w') as f:
        f.write('foobar')
    assert getheader(badfile) is None


def test_standard(fits_file):
    h0 = getheader(fits_file, hdu=0)
    h1 = getheader(fits_file, hdu=1)
    assert isinstance(h0, fits.Header)
    assert isinstance(h1, fits.Header)
    assert not fits.HeaderDiff(h0, h1).identical
