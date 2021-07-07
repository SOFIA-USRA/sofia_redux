# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.spectroscopy.getapertures import get_apertures
from sofia_redux.spectroscopy.tests.resources import rectified_data


@pytest.fixture
def aperture_data():
    rectimg, medprof, spatmap = rectified_data(all_positive=False)
    profile = {}
    apertures = {}
    for order in rectimg:
        profile[order] = np.vstack([rectimg[order]['spatial'],
                                    medprof[order]])
        apertures[order] = [{'position': 14.5,
                             'fwhm': 4.0,
                             'sign': 1},
                            {'position': 80.5,
                             'fwhm': 4.0,
                             'sign': -1}]
    return profile, apertures


@pytest.mark.parametrize('refit,background', [(True, True), (True, False),
                                              (False, True), (False, False)])
def test_radii_fwhm_background(aperture_data, refit, background):
    profile, apertures = aperture_data

    # get two apertures in each of 2 orders
    result = get_apertures(profile, apertures, get_bg=background,
                           refit_fwhm=refit)
    assert len(result) == 2
    for order, apreg in result.items():
        res = apreg['apertures']
        assert len(res) == 2
        for i in range(2):
            fwhm = res[i]['fwhm']
            assert np.allclose(res[i]['position'],
                               apertures[order][i]['position'])
            if refit:
                assert np.allclose(fwhm, apertures[order][i]['fwhm'], atol=0.5)
            else:
                assert np.allclose(fwhm, apertures[order][i]['fwhm'])
            assert res[i]['sign'] == apertures[order][i]['sign']

            assert np.allclose(res[i]['aperture_radius'], fwhm * 0.7)
            assert np.allclose(res[i]['psf_radius'], fwhm * 2.15)

            if background:
                # two regions on either side of two apertures
                # (but may be blended to 3 or expanded to 5,
                # depending on rounding)
                assert 3 <= len(apreg['background']['regions']) <= 5
                # apertures and background do not overlap
                assert not np.any(apreg['background']['mask']
                                  & (res[i]['mask']))
            else:
                assert 'background' not in apreg


def test_fix_aprad(aperture_data):
    profile, apertures = aperture_data

    # assign aperture radius but not psf radius
    for order in apertures:
        for i, ap in enumerate(apertures[order]):
            ap['aperture_radius'] = i + 1.0
    result = get_apertures(profile, apertures,
                           get_bg=False, refit_fwhm=False)
    for order, apreg in result.items():
        res = apreg['apertures']
        for i, ap in enumerate(res):
            fwhm = ap['fwhm']
            assert np.allclose(res[i]['aperture_radius'], i + 1)
            assert np.allclose(res[i]['psf_radius'], fwhm * 2.15)


def test_fix_psfrad(aperture_data):
    profile, apertures = aperture_data

    # assign psf radius but not aperture radius
    for order in apertures:
        for i, ap in enumerate(apertures[order]):
            ap['psf_radius'] = i + 4.0
    result = get_apertures(profile, apertures,
                           get_bg=False, refit_fwhm=False)
    for order, apreg in result.items():
        res = apreg['apertures']
        for i, ap in enumerate(res):
            fwhm = ap['fwhm']
            assert np.allclose(res[i]['aperture_radius'], fwhm * 0.7)
            assert np.allclose(res[i]['psf_radius'], i + 4)


def test_fix_allrad(aperture_data, capsys):
    profile, apertures = aperture_data

    # assign both radii to reasonable values
    for order in apertures:
        for i, ap in enumerate(apertures[order]):
            ap['aperture_radius'] = i + 1.0
            ap['psf_radius'] = i + 4.0
    result = get_apertures(profile, apertures,
                           get_bg=False, refit_fwhm=False)
    for order, apreg in result.items():
        res = apreg['apertures']
        for i, ap in enumerate(res):
            assert np.allclose(res[i]['aperture_radius'], i + 1)
            assert np.allclose(res[i]['psf_radius'], i + 4)

    # assign the PSF radius too high
    for order in apertures:
        for i, ap in enumerate(apertures[order]):
            ap['aperture_radius'] = i + 1.0
            ap['psf_radius'] = i + 20.0
    result = get_apertures(profile, apertures,
                           get_bg=False, refit_fwhm=False)
    # throws warnings
    capt = capsys.readouterr()
    assert 'PSF radius overlaps the low edge' in capt.err
    assert 'PSF radius overlaps the high edge' in capt.err

    for order, apreg in result.items():
        res = apreg['apertures']
        for i, ap in enumerate(res):
            # ap radius is used, psf radius is reduced
            assert np.allclose(res[i]['aperture_radius'], i + 1)
            assert res[i]['psf_radius'] < (i + 20)

    # assign the aperture radius too high
    for order in apertures:
        for i, ap in enumerate(apertures[order]):
            ap['aperture_radius'] = i + 5.0
            ap['psf_radius'] = i + 4.0
    result = get_apertures(profile, apertures,
                           get_bg=False, refit_fwhm=False)
    # throws warnings
    capt = capsys.readouterr()
    assert 'Aperture radius overlaps the PSF radius' in capt.err
    for order, apreg in result.items():
        res = apreg['apertures']
        for i, ap in enumerate(res):
            # psf radius is used for both
            assert np.allclose(res[i]['aperture_radius'], i + 4)
            assert np.allclose(res[i]['psf_radius'], i + 4)


def test_aperture_set_failure(aperture_data, capsys):
    profile, apertures = aperture_data

    # psf radius is too small: no data included
    apertures[1][0]['psf_radius'] = 0.5
    with pytest.raises(ValueError):
        get_apertures(profile, apertures,
                      get_bg=False, refit_fwhm=False)
    assert 'PSF radius too small. Auto-set failed ' \
           'for order 1' in capsys.readouterr().err


def test_aperture_overlap_fail(aperture_data, capsys):
    profile, apertures = aperture_data

    # put apertures on top of each other
    apertures[1][1]['position'] = apertures[1][0]['position']
    with pytest.raises(ValueError):
        get_apertures(profile, apertures,
                      get_bg=False, refit_fwhm=False)
    capt = capsys.readouterr()
    assert 'Apertures overlap' in capt.err
    assert 'Auto-set failed for order 1, ' \
           'aperture center 14.5' in capt.err

    # put apertures close to each other, in the middle
    apertures[1][0]['position'] = 50.0
    apertures[1][1]['position'] = 53.0
    result = get_apertures(profile, apertures,
                           get_bg=False, refit_fwhm=False)
    capt = capsys.readouterr()
    assert 'Apertures overlap' in capt.err

    # aperture and psf radius are reduced for both apertures
    assert np.allclose(result[1]['apertures'][0]['aperture_radius'], 0.1)
    assert np.allclose(result[1]['apertures'][0]['psf_radius'], 0.1)
    assert np.allclose(result[1]['apertures'][1]['aperture_radius'], 0.1)
    assert np.allclose(result[1]['apertures'][1]['psf_radius'], 0.1)

    # put apertures close together, near the top of the slit
    apertures[1][0]['position'] = 94.0
    apertures[1][1]['position'] = 96.0
    with pytest.raises(ValueError):
        get_apertures(profile, apertures,
                      get_bg=False, refit_fwhm=False)
    capt = capsys.readouterr()
    assert 'Apertures overlap' in capt.err
    assert 'Auto-set failed for order 1, ' \
           'aperture center 96' in capt.err
