# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.modeling.models import Gaussian1D
from astropy.modeling.polynomial import Polynomial1D
import numpy as np
import pytest

from sofia_redux.spectroscopy.tracespec import tracespec


@pytest.fixture
def data():
    spatcal, wavecal = np.mgrid[:100, :100]
    spatcal = spatcal.astype(float)
    curve = Polynomial1D(2)
    curve.parameters = [0, -0.1, 0.001]
    dy = curve(np.arange(100))
    model = Gaussian1D(amplitude=2, mean=50, stddev=1.5)  # 2.5 deviation
    image = model(spatcal - dy)

    rectified = {1: {'image': image,
                     'wave': np.arange(100, dtype=float),
                     'spatial': np.arange(100, dtype=float)}}
    positions = {1: [50.0]}

    return rectified, positions


def test_failures(data, capsys):
    rect, positions = data

    # bad order argument -- missing in rect
    info = {}
    result = tracespec(rect, positions, orders=[1, 10, 11],
                       info=info)
    assert len(result) == 1
    assert 1 in result
    assert 1 in info
    capt = capsys.readouterr()
    assert 'Order 10 is missing from rectimg' in capt.err
    assert 'Order 11 is missing from rectimg' in capt.err

    # missing keys in rectified data
    im = rect[1]['image'].copy()
    w = rect[1]['wave'].copy()
    s = rect[1]['spatial'].copy()
    for missing_key in ['image', 'wave', 'spatial']:
        badrect = rect.copy()
        badrect[1] = {'image': im.copy(), 'wave': w.copy(),
                      'spatial': s.copy()}
        del badrect[1][missing_key]
        assert len(tracespec(badrect, positions)) == 0
        assert f'missing {missing_key} key' in capsys.readouterr().err

    # bad order argument -- missing in positions
    info = {}
    rect[10] = rect[1].copy()
    result = tracespec(rect, positions, orders=[1, 10],
                       info=info)
    assert len(result) == 1
    assert 1 in result
    assert 1 in info
    capt = capsys.readouterr()
    assert 'Order 10 is missing from positions' in capt.err
    del rect[10]

    # bad image shape
    rect[1]['image'] = np.zeros(100)
    result = tracespec(rect, positions)
    assert len(result) == 0
    assert 'Invalid image dimensions for order 1' in capsys.readouterr().err


def test_success(data):
    rectified, positions = data
    info = {}
    result = tracespec(rectified, positions, info=info, fast=False)
    assert np.allclose(result[1], [[[50, -0.1, 0.001]]])
    assert set(info.keys()) == {1, 'peak_model'}
    wave = info[1]['wave'][0]
    assert np.allclose(info[1]['trace_model'][0](wave),
                       info[1]['spatial'][0], rtol=1e-3)

    # fast should give same results
    fast = tracespec(rectified, positions, fast=True)
    assert np.allclose(fast[1], result[1])

    # single position (instead of array) -- same
    onepos = tracespec(rectified, {1: 50})
    assert np.allclose(onepos[1], result[1])


def test_model_fit_failure(data, capsys):
    rectified, positions = data

    info = {}
    default = tracespec(rectified, positions, info=info, step=3)
    assert info[1]['mask'][0][2]

    # induce a fit error
    rectified[1]['image'][:, 0:11] = np.nan
    info = {}
    missing_column = tracespec(rectified, positions, info=info)
    assert not info[1]['mask'][0][2]

    # result should be same
    assert np.allclose(default[1], missing_column[1])
    assert 'Invalid data in initial fit' in capsys.readouterr().err
