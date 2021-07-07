# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import time

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.toolkit.utilities.fits import write_hdul


@pytest.fixture
def fits_file(tmpdir):
    filename = str(tmpdir.mkdir('test_robust_read').join('test.fits'))
    hdul = fits.HDUList(fits.PrimaryHDU(data=np.arange(10)))
    hdul.append(fits.ImageHDU(data=np.zeros((10, 10))))
    outfile = os.path.join(os.path.dirname(filename), 'outfile.fits')
    hdul[0].header['FILENAME'] = outfile
    hdul.writeto(filename)
    hdul.close()
    return filename


def test_no_filename(fits_file):
    hdul = fits.open(fits_file)
    outfile = hdul[0].header['FILENAME']
    del hdul[0].header['FILENAME']
    write_hdul(hdul)
    assert not os.path.isfile(outfile)


def test_standard(fits_file):
    hdul = fits.open(fits_file)
    result = write_hdul(hdul)
    assert os.path.isfile(hdul[0].header['FILENAME'])
    assert result == hdul[0].header['FILENAME']


def test_outdir(fits_file):
    hdul = fits.open(fits_file)
    with pytest.raises(ValueError):
        write_hdul(hdul, outdir=1)

    hdul[0].header['FILENAME'] = os.path.basename(hdul[0].header['FILENAME'])
    outdir = os.path.join(os.path.dirname(fits_file), 'newdir')
    outfile = os.path.join(outdir, hdul[0].header['FILENAME'])
    os.mkdir(outdir)
    write_hdul(hdul, outdir=outdir)
    assert os.path.isfile(outfile)


def test_overwrite(fits_file):
    hdul = fits.open(fits_file)
    outfile = hdul[0].header['FILENAME']
    write_hdul(hdul)
    assert os.path.isfile(outfile)

    t1 = os.path.getmtime(outfile)
    time.sleep(0.5)
    write_hdul(hdul, overwrite=False)
    t2 = os.path.getmtime(outfile)
    assert t1 == t2
    write_hdul(hdul, overwrite=True)
    t2 = os.path.getmtime(outfile)
    assert t1 != t2


def test_error(fits_file):
    hdul = fits.open(fits_file)
    with pytest.raises(KeyError):
        hdul[0].data = np.asarray(hdul[0].data, dtype=object)

    assert write_hdul(hdul, overwrite=True) is None
    hdul.close()
