# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.toolkit.utilities.fits import hdinsert


@pytest.fixture
def header():
    header = fits.Header()
    header['KEY_A'] = 1, 'comment_a'
    header['KEY_B'] = 2, 'comment_b'
    header['KEY_C'] = 3, 'comment_c'
    return header


def test_errors():
    assert hdinsert('a', 'TESTKEY', 1) is None  # doesn't raise an error


def test_replace_key(header):
    hdinsert(header, 'KEY_B', -1)
    assert header['KEY_B'] == -1
    assert header.comments['KEY_B'] == 'comment_b'

    hdinsert(header, 'KEY_B', -2, comment='replaced this comment')
    assert header['KEY_B'] == -2
    assert header.comments['KEY_B'] == 'replaced this comment'


def test_new_key(header):
    hdinsert(header, 'NEW_KEY', True)
    assert header['NEW_KEY'] is True
    assert header.comments['NEW_KEY'] == ''
    hdinsert(header, 'NEW_KEY2', 1.5, comment='this is a new key')
    assert header['NEW_KEY2'] == 1.5
    assert header.comments['NEW_KEY2'] == 'this is a new key'


def test_insert_after(header):
    hdinsert(header, 'NEW_KEY', 'new_value', after=True, refkey='KEY_A',
             comment='this is an inserted key')
    assert list(header.keys())[1] == 'NEW_KEY'
    assert list(header.keys())[2] == 'KEY_B'

    assert header[1] == 'new_value'
    assert header.comments['NEW_KEY'] == 'this is an inserted key'
