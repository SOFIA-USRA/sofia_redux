# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.spectroscopy.readspec import readspec


@pytest.fixture
def data(tmpdir):
    header = fits.header.Header()
    expected = {}
    header['SLTH_ARC'] = 1.2
    header['SLTH_PIX'] = 1.3
    header['SLTW_ARC'] = 1.4
    header['SLTW_PIX'] = 1.5
    header['RP'] = 999
    header['MODENAME'] = 'testmode'
    header['NORDERS'] = 2
    header['NAPS'] = 3
    header['INSTR'] = 'testinst'
    header['START'] = 10
    header['STOP'] = 20
    header['AIRMASS'] = 1.0
    header['XUNITS'] = 'um'
    header['YUNITS'] = 'Jy/pixel'
    header['RAWUNITS'] = 'Me/s'

    expected.update(header)
    expected['SLITH_ARC'] = expected.pop('SLTH_ARC')
    expected['SLITH_PIX'] = expected.pop('SLTH_PIX')
    expected['SLITW_ARC'] = expected.pop('SLTW_ARC')
    expected['SLITW_PIX'] = expected.pop('SLTW_PIX')
    expected['OBSMODE'] = expected.pop('MODENAME')
    expected['RUNITS'] = expected.pop('RAWUNITS')

    header['ORDERS'], expected['ORDERS'] = '1,2', [1, 2]
    header['BGR'] = '0-10,20-30;5-8,10-12'
    expected['BGR'] = [[[0, 10], [20, 30]], [[5, 8], [10, 12]]]

    header['APPOSO01'] = '15,32,40'
    header['APRADO01'] = '2,2,2'
    header['PSFRAD01'] = '3,3,3'
    header['APPOSO02'] = '2,7,15'
    header['APRADO02'] = '1,1,1'
    header['PSFRAD02'] = '2,2,5'

    expected['APPOS'] = [[15, 32, 40], [2, 7, 15]]
    expected['APRAD'] = [[2, 2, 2], [1, 1, 1]]
    expected['PSFRAD'] = [[3, 3, 3], [2, 2, 5]]

    image = np.arange(2 * 3 * 5 * 100, dtype=float).reshape(6, 5, 100)
    expected['spectra'] = image.reshape(2, 3, 5, 100)

    hdul = fits.HDUList(fits.PrimaryHDU(image, header=header))
    filename = str(tmpdir.join('default.fits'))
    hdul.writeto(filename)

    return filename, expected


def test_failure(tmpdir, capsys):
    # missing file
    assert readspec('__not_a_file__') is None
    assert 'not a file' in capsys.readouterr().err

    # bad header
    fname = str(tmpdir.join('badfile.fits'))
    hdul = fits.HDUList(fits.PrimaryHDU())
    hdul.writeto(fname)
    assert readspec(fname) is None
    assert 'first HDU has no data' in capsys.readouterr().err


def test_success(data):
    filename, expected = data
    result = readspec(filename)
    for k, v in expected.items():
        if isinstance(v, (list, np.ndarray)):
            if len(v) != 0:
                assert np.allclose(result[k.lower()], v)
            else:
                assert len(result[k.lower()]) == 0
        else:
            assert result[k.lower()] == v


def test_onespec(data, capsys):
    filename, expected = data

    # modify data to be single order/aperture style
    hdul = fits.open(filename)
    hdul[0].data = hdul[0].data[0]

    # raises error if norders/naps doesn't match
    with pytest.raises(ValueError):
        readspec(hdul)
    assert 'Mismatch: data size' in capsys.readouterr().err
    hdul[0].header['NORDERS'] = 1
    with pytest.raises(ValueError):
        readspec(hdul)
    assert 'Mismatch: data size' in capsys.readouterr().err
    hdul[0].header['NAPS'] = 1

    # also if apertures don't match
    with pytest.raises(ValueError):
        readspec(hdul)
    assert 'Mismatch: data size' in capsys.readouterr().err
    hdul[0].header['APPOSO01'] = '15'
    hdul[0].header['APRADO01'] = '2'
    hdul[0].header['PSFRAD01'] = '4'

    # okay with correct headers
    result = readspec(hdul)
    assert result['spectra'].shape == (1, 1, *expected['spectra'][0][0].shape)
    assert np.allclose(result['spectra'][0][0], expected['spectra'][0][0])

    # but not if data is not at least 2d
    hdul[0].data = hdul[0].data[0]
    with pytest.raises(ValueError):
        readspec(hdul)
    assert 'Invalid data shape' in capsys.readouterr().err
