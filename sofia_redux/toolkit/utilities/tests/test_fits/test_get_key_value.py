# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits

from sofia_redux.toolkit.utilities.fits import get_key_value


def test_strings():
    header = fits.Header()
    header['TEST'] = '  abcdef '
    assert get_key_value(header, 'TEST') == 'ABCDEF'


def test_nonstring():
    header = fits.Header()
    header['TEST'] = 1.0
    result = get_key_value(header, 'TEST')
    assert isinstance(result, float)
    assert result == 1


def test_defaults():
    header = fits.Header()
    header['TEST'] = 1
    assert get_key_value(header, 'INVALID', default=3) == 3
