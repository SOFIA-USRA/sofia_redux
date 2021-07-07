# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.toolkit.utilities.fits \
    import hdinsert, add_history, kref, href


@pytest.fixture
def header():
    header = fits.Header()
    header['KEY_A'] = 'a'
    header['KEY_B'] = 'b'
    header[kref] = ''
    header['KEY_C'] = 'c'
    header['KEY_D'] = 'd'
    hdinsert(header, href, '', refkey='HISTORY', after=True)
    header['HISTORY'] = 'old history 1'
    header['HISTORY'] = 'old history 2'
    return header


def test_add_history(header):
    add_history(header, 'new history 1')
    add_history(header, 'new history 2')
    assert header[6] == 'old history 1'
    assert header[7] == 'old history 2'
    assert header[8] == 'new history 1'
    assert header[9] == 'new history 2'


def test_prefix(header):
    add_history(header, 'new history', prefix='test prefix')
    assert header[8] == 'test prefix: new history'


def test_no_href(header):
    del header[5]
    add_history(header, 'new history 1')
    assert header[5] == 'old history 1'
    assert header[6] == 'old history 2'
    assert header[7] == 'new history 1'
