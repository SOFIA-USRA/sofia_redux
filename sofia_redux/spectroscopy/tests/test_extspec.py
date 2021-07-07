# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.spectroscopy.extspec import extspec
from sofia_redux.spectroscopy.tests.resources import rectified_data


@pytest.fixture
def data():
    return rectified_data()


def test_failure(data, capsys):
    """Test failure cases."""
    rectimg, medprof, spatmap = data

    # optimal without profiles fails
    assert extspec(rectimg, optimal=True) is None
    assert 'requires spatial map' in capsys.readouterr().err

    # mismatch between apsign and apmask fails
    rectimg[1]['apsign'] = [1, -1, 1]
    assert extspec(rectimg) is None
    assert 'Mismatch' in capsys.readouterr().err

    # reset
    del rectimg[1]
    capsys.readouterr()

    # optimal extraction fails at each column if no good data
    # in image or profile --
    # will warn only, once for each aperture at each wavelength,
    # and add 8 to the bitmask
    medprof[2] *= np.nan
    nx = rectimg[2]['image'].shape[1]
    result = extspec(rectimg, sub_background=False,
                     optimal=True, profile=medprof)
    assert capsys.readouterr().err.count(
        'Optimal extraction failed') == nx * 2
    assert np.all(result[2][:, 3] == 9)


@pytest.mark.parametrize('optimal', [True, False])
def test_extraction(data, optimal):
    """Test basic extraction on synthetic data."""

    # results should be the same for standard and optimal
    rectimg, medprof, spatmap = data
    result = extspec(rectimg, profile=medprof, optimal=optimal,
                     threshold=0, bgorder=0)

    # orders are stored as keys
    assert np.allclose(np.unique(list(result.keys())), [1, 2])

    # check extracted spectra for each order
    for order, spec in result.items():
        # 2 apertures, 4 data rows (wave, flux, error, bitmask)
        assert spec.shape[0] == 2 and spec.shape[1] == 4

        # wavecal
        if order == 1:
            # first half of array
            assert np.all(spec[:, 0] <= 50)
        else:
            # second half of array
            assert np.all(spec[:, 0] > 50)

        # flux: values are  22233433222, so total flux should be around 6,
        # with background subtracted
        assert np.allclose(spec[:, 1], 6, atol=1)

        # error: values are 111111111111, with aperture radius
        # including 4-5 pixels, so value should be ~5/sqrt(5) = 2.2
        assert np.allclose(spec[:, 2], 2.2, atol=1)

        # bitmask is all 1
        assert np.allclose(spec[:, 3], 1)


@pytest.mark.parametrize('profile', [True, False])
def test_fixdata(data, profile):
    """Test fix_data cases."""
    rectimg, medprof, spatmap = data

    # set some NaNs in the first aperture, first order data
    rectimg[1]['image'][13, :] = np.nan
    sum_nan = np.sum(np.isnan(rectimg[1]['image']))

    # kwargs for profile or spatial map
    kwargs = {'optimal': False,
              'threshold': 0, 'bgorder': 0}
    if profile:
        kwargs['profile'] = medprof
    else:
        kwargs['spatial_map'] = spatmap

    # without fix_bad, nans are still there in the result
    unfixed = extspec(rectimg, fix_bad=False, **kwargs)
    assert np.sum(np.isnan(rectimg[1]['image'])) == sum_nan

    # with fix_bad, they're gone
    fixed = extspec(rectimg, fix_bad=True, **kwargs)
    assert np.sum(np.isnan(rectimg[1]['image'])) == 0

    # flux with fix_bad should be higher than without
    assert np.all(unfixed[1][0, 1] < fixed[1][0, 1])

    # bit mask should have 2 added for all columns
    # for first aperture only
    assert np.allclose(fixed[1][0, 3], 3)
    assert np.allclose(fixed[1][1, 3], 1)


def test_masks(data):
    """Test badmask and bitmask cases."""
    rectimg, medprof, spatmap = data

    # just do one order
    del rectimg[2]

    kwargs = {'profile': medprof, 'fix_bad': True,
              'threshold': 0, 'bgorder': 0}

    # default: mask empty, bitmask=1
    result = extspec(rectimg, **kwargs)
    assert np.allclose(result[1][:, 3], 1)

    # set a bad pixel in every column in the mask: marked in obit
    rectimg[1]['mask'][13, :] = False
    result = extspec(rectimg, **kwargs)
    assert np.allclose(result[1][0, 3], 3)

    # no mask provided: same as default
    del rectimg[1]['mask']
    rectimg[1]['bitmask'][:] = 1
    result = extspec(rectimg, **kwargs)
    assert np.allclose(result[1][:, 3], 1)

    # no mask or bitmask provided: all zero
    del rectimg[1]['bitmask']
    result = extspec(rectimg, **kwargs)
    assert np.allclose(result[1][:, 3], 0)


def test_bgsub(data, capsys, mocker):
    """Test background subtraction cases."""
    rectimg, medprof, spatmap = data

    # order 1 only for simplicity
    del rectimg[2]
    img_copy = rectimg[1]['image'].copy()
    apmask_copy = rectimg[1]['apmask'].copy()

    kwargs = {'threshold': 0, 'bgorder': 0, 'optimal': False}

    # default: subtract background from image and spectrum
    result = extspec(rectimg, sub_background=True, **kwargs)
    assert np.allclose(np.median(rectimg[1]['image']), 0)
    assert np.allclose(result[1][:, 1], 6, atol=0.1)

    # don't subtract background: should stay at ~2, total flux is higher
    rectimg[1]['image'] = img_copy.copy()
    result = extspec(rectimg, sub_background=False, **kwargs)
    assert np.allclose(np.median(rectimg[1]['image']), 2)
    assert np.allclose(result[1][:, 1], 24, atol=1)

    # try to subtract background without BG regions defined:
    # warns and turns off background subtraction
    rectimg[1]['image'] = img_copy.copy()
    apmask = rectimg[1]['apmask']
    apmask[np.isnan(apmask)] = 0
    result = extspec(rectimg, sub_background=True, **kwargs)
    assert np.allclose(np.median(rectimg[1]['image']), 2)
    assert np.allclose(result[1][:, 1], 24, atol=1)
    assert 'No background regions found' in capsys.readouterr().err

    # background failure for too few points: warns for each column
    apmask[40, :] = np.nan
    result = extspec(rectimg, sub_background=True, **kwargs)
    assert np.allclose(np.median(rectimg[1]['image']), 2)
    assert np.allclose(result[1][:, 1], 24, atol=1)
    nx = rectimg[1]['image'].shape[1]
    capt = capsys.readouterr()
    assert capt.err.count('Background fit failed') == nx
    assert 'Not enough background points' in capt.err

    # failure in polynomial fit: same effect
    class MockModel(object):
        def __init__(self):
            self.success = False

    mocker.patch('sofia_redux.spectroscopy.extspec.polyfitnd',
                 return_value=MockModel())

    rectimg[1]['apmask'] = apmask_copy.copy()
    result = extspec(rectimg, sub_background=True, **kwargs)
    assert np.allclose(np.median(rectimg[1]['image']), 2)
    assert np.allclose(result[1][:, 1], 24, atol=1)
    capt = capsys.readouterr()
    assert capt.err.count('Background fit failed') == nx
    assert 'Polynomial fit failed' in capt.err


def test_sum_function(data):
    """Test alternate sum function for standard extraction."""

    rectimg, medprof, spatmap = data
    default = extspec(rectimg, optimal=False, threshold=0, bgorder=0)

    def err_func(var, weights):
        var[weights < 1] = np.nan
        count = np.sum(~np.isnan(var))
        return np.sqrt(np.nansum(var)) / count

    def sum_func(flux, weights):
        flux[weights < 1] = np.nan
        return np.nanmean(flux)

    meanspec = extspec(rectimg, optimal=False, threshold=0, bgorder=0,
                       sum_function=sum_func, error_function=err_func)

    # check extracted spectra for each order
    for order, spec in default.items():
        # wavecal, bitmask should be same as default
        assert np.allclose(spec[:, 0], meanspec[order][:, 0])
        assert np.allclose(spec[:, 3], meanspec[order][:, 3])

        # flux: bg subbed values are [0,0,0.5,1,1.5,1.5,1,0.5,0],
        # so mean should be 2/3
        assert np.allclose(meanspec[order][:, 1], 2 / 3)

        # error: values are all ~1.1 so should be sqrt(1.1)/sqrt(9)
        assert np.allclose(meanspec[order][:, 2], 0.35, atol=0.1)
