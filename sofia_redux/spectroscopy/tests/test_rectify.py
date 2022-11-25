# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.spectroscopy.rectify import rectify


@pytest.fixture
def data():
    spatcal, wavecal = np.mgrid[:64, :64]
    spatcal, wavecal = spatcal.astype(float), wavecal.astype(float)
    ordermask = np.full((64, 64), 0)
    ordermask[1:63, :32] = 1
    ordermask[1:63, 32:] = 2
    image = np.full((64, 64), 2.0)
    return image, ordermask, wavecal, spatcal


def test_failure(data):
    image, ordermask, wavecal, spatcal = data
    assert rectify(image, ordermask[0], wavecal, spatcal) is None
    assert rectify(image, ordermask * 0, wavecal, spatcal) is None


def test_success(data):
    image, ordermask, wavecal, spatcal = data
    result = rectify(image, ordermask, wavecal, spatcal)
    assert np.allclose(np.unique(list(result.keys())), [1, 2])
    for r in result.values():
        for key in ['image', 'mask', 'pixsum', 'spatial',
                    'wave', 'variance', 'bitmask']:
            assert key in r
    result = rectify(image, ordermask, wavecal, spatcal, orders=1)
    keys = list(result.keys())
    assert len(keys) == 1 and keys[0] == 1


def test_header(data):
    image, ordermask, wavecal, spatcal = data
    header = fits.Header({'TEST': 'VALUE'})
    result = rectify(image, ordermask, wavecal, spatcal, header=header)

    # input header is copied for each order
    for order in result:
        assert result[order]['header'] is not header
        assert result[order]['header']['TEST'] == 'VALUE'
