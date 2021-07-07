# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.spectroscopy.mkapmask import mkapmask
from sofia_redux.spectroscopy.tests.resources import rectified_data


@pytest.fixture
def aperture_data():
    rectimg, medprof, spatmap = rectified_data(all_positive=False)
    apertures = [{'position': 14.5, 'fwhm': 4.0, 'sign': 1,
                  'trace': np.full(rectimg[1]['wave'].size, 14.5),
                  'psf_radius': 4., 'aperture_radius': 2.},
                 {'position': 80.5, 'fwhm': 4.0, 'sign': -1,
                  'trace': np.full(rectimg[1]['wave'].size, 80.5),
                  'psf_radius': 4., 'aperture_radius': 2.}]
    slit = rectimg[1]['spatial']
    wave = rectimg[1]['wave']
    background = [[25, 38], [54, 71]]

    return slit, wave, apertures, background


def test_success(aperture_data):
    slit, wave, apertures, background = aperture_data
    mask = mkapmask(slit, wave, apertures, background=background)

    slitcoord = np.column_stack([slit] * mask.shape[1])
    seen = np.full(mask.shape, False)

    # background
    xs = ((slitcoord >= 25) & (slitcoord <= 38)) \
        | ((slitcoord >= 54) & (slitcoord <= 71))
    assert np.all(np.isnan(mask[xs]))
    seen[xs] = True

    # psf radius
    ap1 = (np.abs(mask) > 0) & (np.abs(mask) <= 1)
    assert slitcoord[ap1].min() == 10 and slitcoord[ap1].max() == 19
    seen[ap1] = True
    ap2 = (np.abs(mask) > 1) & (np.abs(mask) <= 2)
    assert slitcoord[ap2].min() == 76 and slitcoord[ap2].max() == 85
    seen[ap2] = True

    # check partial pixels
    assert np.allclose(mask[slitcoord == 10], 0.5)
    assert np.allclose(mask[slitcoord == 19], 0.5)
    assert np.allclose(mask[slitcoord == 76], 1.5)
    assert np.allclose(mask[slitcoord == 85], 1.5)

    # unused
    assert np.allclose(mask[~seen], 0)

    # ap radius
    ap1 = (mask == -1)
    assert np.allclose(slitcoord[ap1].min(), 13)
    assert np.allclose(slitcoord[ap1].max(), 16)
    ap2 = (mask == -2)
    assert np.allclose(slitcoord[ap2].min(), 79)
    assert np.allclose(slitcoord[ap2].max(), 82)


def test_no_aprad(aperture_data):
    slit, wave, apertures, background = aperture_data

    # take out aperture radius info
    for ap in apertures:
        del ap['aperture_radius']

    mask = mkapmask(slit, wave, apertures, background=background)

    # no negative values in output
    assert np.all(np.isnan(mask) | (mask >= 0))


def test_overlap(aperture_data, capsys):
    slit, wave, apertures, background = aperture_data

    # increase the psf radius to make apertures overlap
    for ap in apertures:
        ap['psf_radius'] = 40.

    with pytest.raises(ValueError):
        mkapmask(slit, wave, apertures)
    assert 'extraction apertures overlap' in capsys.readouterr().err


def test_single_slitval(aperture_data):
    slit, wave, apertures, background = aperture_data

    # take only one slit/aperture value
    slit = 14.5
    apertures = [apertures[0]]

    mask = mkapmask(slit, wave, apertures)
    assert mask.shape == (1, wave.size)
    assert np.all(mask == -1)
