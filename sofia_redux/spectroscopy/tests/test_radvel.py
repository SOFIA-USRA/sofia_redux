# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.spectroscopy.radvel import radvel


@pytest.fixture
def header():
    header = fits.header.Header()
    header['DATE-OBS'] = '2016-03-01T10:38:39'
    header['TELRA'] = 15.7356
    header['TELDEC'] = -18.4388
    header['ALTI_STA'] = 41000.
    header['ALTI_END'] = 41000.
    header['ZA_START'] = 45.
    header['ZA_END'] = 45.
    header['LAT_STA'] = 40.
    header['LON_STA'] = -120.
    return header


def test_success(header):
    rv1 = radvel(header)

    assert np.isclose(rv1[0], 9.86044e-5)
    assert np.isclose(rv1[1], 3.43838e-5)
    assert np.isclose(rv1[0] + rv1[1], 0.00013299)

    # test with a different equinox -- should be close,
    # but not the same
    rv2 = radvel(header, equinox='J1900')
    assert not np.allclose(rv2, rv1, atol=1e-8)
    assert np.allclose(rv2, rv1, atol=1e-5)


def test_errors(capsys):
    # invalid header
    assert radvel([]) is None
    capt = capsys.readouterr()
    assert 'Invalid header' in capt.err

    # missing keys
    header = fits.Header()
    for key in ['DATE-OBS', 'TELRA', 'TELDEC',
                'LAT_STA', 'LON_STA', 'ALTI_STA']:
        assert radvel(header) is None
        capt = capsys.readouterr()
        assert '{} not found'.format(key) in capt.err
        header[key] = 'TESTVAL'

    # bad dateobs
    assert radvel(header) is None
    capt = capsys.readouterr()
    assert 'Unable to convert' in capt.err
