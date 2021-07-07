# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.toolkit.utilities.fits \
    import hdinsert, add_history_wrap, kref, href


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


def test_add_history_wrap(header):
    add_history = add_history_wrap('test procedure')
    add_history(header, 'test history')
    assert header[8] == 'test procedure: test history'
