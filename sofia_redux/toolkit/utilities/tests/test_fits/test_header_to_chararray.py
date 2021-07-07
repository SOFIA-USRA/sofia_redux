# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np

from sofia_redux.toolkit.utilities.fits import header_to_chararray


def test_bad_header():
    assert header_to_chararray(1) is None


def test_standard():
    header = fits.Header()
    header['TEST_A'] = 1
    header['TEST_B'] = 2
    header['TEST_C'] = 3
    result = header_to_chararray(header)
    assert isinstance(result, np.chararray)
    assert result.shape == (1, 3)
    assert result[0, 0] == 'TEST_A  =                    1'
    assert result[0, 1] == 'TEST_B  =                    2'
    assert result[0, 2] == 'TEST_C  =                    3'
