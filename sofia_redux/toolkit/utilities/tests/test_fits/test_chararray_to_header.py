# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import chararray_to_header


def test_bad_header():
    assert chararray_to_header(1) is None
    assert chararray_to_header(np.empty((3, 3, 3))) is None


def test_standard():
    chararray = np.array([[
        'TEST_A  =                    1 / comment a',
        'TEST_B  =                    2',
        'TEST_C  =                    3']])

    result = chararray_to_header(chararray)
    assert isinstance(result, fits.Header)
    assert result['TEST_A'] == 1
    assert result.comments['TEST_A'] == 'comment a'
    assert result['TEST_B'] == 2
    assert result.comments['TEST_B'] == ''


def test_error():
    chararray = np.array([[
        'TEST_A  =                    1 / comment a',
        'TEST_B  =                    2',
        'TEST_C  =                    3']])
    chararray = np.asarray(chararray, dtype=object)
    chararray[0, 2] = 1
    result = chararray_to_header(chararray)
    assert result is None
