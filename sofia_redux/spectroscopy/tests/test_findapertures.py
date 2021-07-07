# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.spectroscopy.findapertures import find_apertures
from sofia_redux.spectroscopy.tests.resources import rectified_data


@pytest.fixture
def profile_data():
    rectimg, medprof, spatmap = rectified_data(all_positive=False)
    profile = {}
    for order in rectimg:
        profile[order] = np.vstack([rectimg[order]['spatial'],
                                    medprof[order]])
    return profile


def test_default(profile_data):
    # find one aperture in each of 2 orders

    # expected position, FWHM, sign for each order
    # centers are at index 80; 0.5 added for middle of pixel calibration
    expected = [80.5, 4, -1]

    result = find_apertures(profile_data)
    assert len(result) == 2
    for order, res in result.items():
        assert len(res) == 1
        assert np.allclose(res[0]['position'], expected[0], atol=0.01)
        assert np.allclose(res[0]['fwhm'], expected[1], atol=1)
        assert res[0]['sign'] == expected[2]


def test_multi_ap(profile_data):
    # find two apertures in each of 2 orders
    # centers are at index 14, 80; 0.5 added for middle of pixel calibration
    expected = [[14.5, 80.5], [4, 4], [1, -1]]

    result = find_apertures(profile_data, npeaks=2)
    assert len(result) == 2
    for order, res in result.items():
        assert len(res) == 2

        assert np.allclose(res[0]['position'], expected[0][0], atol=0.01)
        assert np.allclose(res[0]['fwhm'], expected[1][0], atol=1)
        assert res[0]['sign'] == expected[2][0]

        assert np.allclose(res[1]['position'], expected[0][1], atol=0.01)
        assert np.allclose(res[1]['fwhm'], expected[1][1], atol=1)
        assert res[1]['sign'] == expected[2][1]


def test_one_order(profile_data):
    # find one aperture in one order
    expected = [80.5, 4, -1]
    result = find_apertures(profile_data, orders=[2])
    assert len(result) == 1
    for order, res in result.items():
        assert order == 2
        assert len(res) == 1
        assert np.allclose(res[0]['position'], expected[0], atol=0.01)
        assert np.allclose(res[0]['fwhm'], expected[1], atol=1)
        assert res[0]['sign'] == expected[2]


def test_guess(profile_data):
    # guess near smaller peak for order 1, larger for order 2
    expected = [[14.5, 80.5], [4, 4], [1, -1]]
    positions = {1: [12], 2: [75]}
    result = find_apertures(profile_data, positions=positions, fix=False)
    assert len(result) == 2
    for order, res in result.items():
        assert len(res) == 1
        assert np.allclose(res[0]['position'],
                           expected[0][order - 1], atol=0.01)
        assert np.allclose(res[0]['fwhm'],
                           expected[1][order - 1], atol=1)
        assert res[0]['sign'] == expected[2][order - 1]


def test_fix_with_guess(profile_data):
    # fix near smaller and larger peaks, for both orders
    expected = [[16.0, 79.0], [4, 4], [1, -1]]
    positions = {1: [16, 79], 2: [16, 79]}
    result = find_apertures(profile_data, positions=positions,
                            fix=True, fwhm=4.)
    assert len(result) == 2
    for order, res in result.items():
        assert len(res) == 2

        assert np.allclose(res[0]['position'], expected[0][0])
        assert np.allclose(res[0]['fwhm'], expected[1][0])
        assert res[0]['sign'] == expected[2][0]

        assert np.allclose(res[1]['position'], expected[0][1])
        assert np.allclose(res[1]['fwhm'], expected[1][1])
        assert res[1]['sign'] == expected[2][1]


def test_fix_no_guess(profile_data):
    # fix three apertures, evenly dividing the slit
    expected = [[28, 51, 74], [1, 1, 1]]
    result = find_apertures(profile_data, npeaks=3, positions=None, fix=True)
    assert len(result) == 2
    for order, res in result.items():
        assert len(res) == 3
        for i, ap in enumerate(res):
            # position may be off by 1, depending on the rounding
            assert np.allclose(ap['position'], expected[0][i], atol=1)
            assert np.allclose(ap['fwhm'], expected[1][i])
