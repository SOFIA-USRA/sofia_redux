# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.spectroscopy.mkspatprof import mkspatprof


@pytest.fixture
def data():
    image = np.full((100, 100), 2.0)
    image[50] = 3
    image[51] = 4
    image[52] = 5
    wave = np.arange(100) * 1.0
    spatial = wave * 0.5
    rect = {1: {'image': image, 'wave': wave, 'spatial': spatial},
            2: {'image': image * 2, 'wave': wave + 100,
                'spatial': spatial + 100}}
    atran = np.array([np.arange(200) * 1.0, np.full(200, 1.0)])
    atran[1, 50:100] = 0.1

    rect[1]['image'][0, 50:] = 0
    return rect, atran


def test_failure(data, capsys):
    rect, atran = data

    # bad shape for atran
    assert mkspatprof(rect, atran=np.arange(10), atmosthresh=0.1) is None
    assert 'Invalid atran shape' in capsys.readouterr().err

    # bad order argument
    result = mkspatprof(rect, orders=[2, 10, 11])
    assert len(result) == 1
    assert 2 in result
    capt = capsys.readouterr()
    assert 'Order 10 is missing' in capt.err
    assert 'Order 11 is missing' in capt.err

    # missing keys in rectified data
    im = rect[1]['image'].copy()
    w = rect[1]['wave'].copy()
    s = rect[1]['spatial'].copy()
    for missing_key in ['image', 'wave', 'spatial']:
        badrect = rect.copy()
        badrect[1] = {'image': im.copy(), 'wave': w.copy(),
                      'spatial': s.copy()}
        badrect[2] = {'image': im.copy(), 'wave': w.copy(),
                      'spatial': s.copy()}
        del badrect[1][missing_key]
        del badrect[2][missing_key]
        assert len(mkspatprof(badrect)) == 0
        assert f'missing {missing_key} key' in capsys.readouterr().err


def test_success(data):
    rect, atran = data

    result = mkspatprof(rect, robust=0)
    assert np.allclose(np.unique(list(result.keys())), [1, 2])

    # bad pixel from zeros in 1st order image
    badval = result[1][0]
    assert not np.allclose(badval, 0) and badval < 0
    goodval = result[2][0]
    assert np.allclose(goodval, 0)

    # test background subtraction
    mask = np.full(result[1].size, True)
    mask[0] = False
    mask[50:53] = False
    assert np.allclose(result[1][mask], 0)
    assert np.argmax(result[1]) == 52

    # test profile scaling on the order with no bad pixels
    assert np.allclose(np.sum(result[2]), 1.0)

    # test background subtraction off
    result = mkspatprof(rect, robust=0, bgsub=False)
    assert not np.allclose(result[1][mask], 0)


def test_atran(data):
    rect, atran = data

    medprof, spatmap = mkspatprof(rect, atran=atran, atmosthresh=0.2,
                                  robust=0, return_fit_profile=True)

    # bad region does not affect spatmap
    assert np.allclose(spatmap[1][:, :50], spatmap[1][:, 50:])

    # medprof has no badpix
    badval = medprof[1][0]
    assert np.allclose(badval, 0)

    # without atran correction: atmosthresh=None
    medprof, spatmap = mkspatprof(rect, atran=atran, atmosthresh=None,
                                  robust=0, return_fit_profile=True)
    assert not np.allclose(medprof[1][0], 0)
    assert not np.allclose(spatmap[1][:, :50], spatmap[1][:, 50:])
