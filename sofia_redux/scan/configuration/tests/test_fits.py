# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import log
from astropy.io import fits
from configobj import ConfigObj
import os
import pytest

from sofia_redux.scan.configuration.fits import FitsOptions


@pytest.fixture
def fits_header():
    h = fits.Header()
    h['INTKEY'] = 1, 'An integer value'
    h['FLOATKEY'] = 2.0, 'A float value'
    h['STRKEY'] = 'foo', 'A string value'
    h['PRESERV'] = 'preserve_me', 'A preserved test value'
    return h


@pytest.fixture
def fits_file(tmpdir, fits_header):
    h = fits_header
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU(header=h))
    filename = str(tmpdir.mkdir('test_config').join('test_fits.fits'))
    hdul.writeto(filename, overwrite=True)
    return filename


@pytest.fixture
def fits_options(fits_file):
    """
    Return a test FitsOptions object.

    Parameters
    ----------
    fits_file : str
        The path to a test FITS file.

    Returns
    -------
    FitsOptions
    """
    f = FitsOptions()
    f.update_header(fits_file)
    f.set_preserved_card('PRESERV')
    return f


def test_init():
    f = FitsOptions(allow_error=False, verbose=False)
    assert not f.allow_error and not f.verbose
    f = FitsOptions(allow_error=True, verbose=True)
    assert f.allow_error and f.verbose


def test_copy(fits_options):
    f = fits_options
    f2 = f.copy()
    assert f.options == f2.options and f.options is not f2.options
    assert f.header == f2.header and f.header is not f2.header
    assert f.filename == f2.filename
    assert f.extension == f2.extension
    assert f.preserved_cards == f2.preserved_cards
    assert f.preserved_cards is not f2.preserved_cards


def test_clear(fits_options):
    f = fits_options
    f.clear()
    assert f.size == 0
    assert f.header is None
    assert f.filename is None
    assert f.extension is None
    assert len(f.preserved_cards) == 0


def test_get_item(fits_options):
    f = fits_options
    assert f['INTKEY'] == '1'
    assert f['FLOATKEY'] == '2.0'
    assert f['STRKEY'] == 'foo'
    assert f['PRESERV'] == 'preserve_me'
    with pytest.raises(KeyError) as err:
        _ = f['does_not_exist']
    assert 'is not a valid key' in str(err.value)


def test_set_item(fits_options):
    f = fits_options
    f['new_key'] = 'abc'
    assert 'new_key' in f.options
    assert 'new_key' not in f.header


def test_contains(fits_options):
    f = fits_options
    assert 'INTKEY' in f
    assert 'missing' not in f


def test_read(fits_header, fits_file):
    f = FitsOptions()
    with pytest.raises(ValueError) as err:
        _ = f.read(1)
    assert 'Header must be' in str(err.value)

    options = f.read(fits_header)
    assert options == {'INTKEY': '1', 'FLOATKEY': '2.0',
                       'STRKEY': 'foo', 'PRESERV': 'preserve_me'}
    expected = options.copy()

    filename = fits_file
    test_directory = os.path.dirname(filename)
    bad_file = os.path.join(test_directory, 'a_bad_file.fits')
    with pytest.raises(ValueError) as err:
        f.read(bad_file)
    assert 'Not a valid file' in str(err.value)

    with pytest.raises(Exception):
        with log.log_to_list() as log_list:
            f.read(filename, extension=2)
    assert "Could not read header in extension" in log_list[0].msg

    options = f.read(filename)
    for key, value in expected.items():
        assert options[key] == value


def test_reread(fits_options):
    f = fits_options
    f.options = ConfigObj()
    f.reread()
    assert f.options['INTKEY'] == '1'

    f.options = ConfigObj()
    f.header = None
    f.reread()
    assert len(f.options) == 0


def test_get(fits_options):
    f = fits_options
    assert f.get('dne', default='default') == 'default'
    assert f.get('INTKEY') == '1'


def test_set(fits_options):
    f = fits_options
    f.set('new_key ', 'new_value')
    assert f['new_key'] == 'new_value'


def test_update_header(fits_header):
    f = FitsOptions()
    f.update_header(fits_header)
    assert f['INTKEY'] == '1'


def test_keys(fits_options):
    f = fits_options
    keys = f.keys()
    for key in ['INTKEY', 'FLOATKEY', 'STRKEY', 'PRESERV']:
        assert key in keys


def test_set_preserved_card(fits_options):
    f = fits_options
    assert f.preserved_cards == {
        'PRESERV': ('preserve_me', 'A preserved test value')}

    f.set_preserved_card('INTKEY')
    assert f.preserved_cards == {
        'PRESERV': ('preserve_me', 'A preserved test value'),
        'INTKEY': (1, 'An integer value')}
    current = f.preserved_cards.copy()

    f.set_preserved_card('dne')
    assert f.preserved_cards == current

    f.header = None
    f.set_preserved_card('STRKEY')
    assert f.preserved_cards == current


def test_reset_preserved_cards(fits_options):
    f = fits_options
    f.reset_preserved_cards()
    assert len(f.preserved_cards) == 0
