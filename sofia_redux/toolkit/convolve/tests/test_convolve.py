# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from scipy.spatial import Delaunay

from sofia_redux.toolkit.convolve.base import ConvolveBase
from sofia_redux.toolkit.utilities.func import stack


@pytest.fixture
def data2d():
    y, x = np.mgrid[:50, :50]
    y = y.astype(float)
    mask = np.full(y.shape, True)
    mask[25, 25] = False
    z = np.full(y.shape, 2.0)
    z[25, 25] = np.nan
    samples = stack(x, y, z)

    return samples, mask.ravel()


def test_properties():
    x = np.arange(100).astype(float)
    y = (x % 2) - 0.5
    e = np.full(x.size, 2.0)
    s = ConvolveBase(x, y, error=e)
    s._interpolated_error = None
    assert s.error.size == x.size
    assert np.allclose(s.error, 2)

    s = ConvolveBase(x, y, error=2)
    s._interpolated_error = None
    assert s.error.size == x.size
    assert np.allclose(s.error, 2)

    assert s.residuals.size == x.size
    assert np.allclose(s.residuals, 0)
    s.stats = None
    assert s.residuals is None

    assert s.masked.size == x.size
    assert s.masked.all()


def test_replaced_masked_samples(data2d):
    x = np.arange(100).astype(float)
    y = x * 2
    samples = np.stack((x, y))
    mask = np.full(x.size, True)

    r = ConvolveBase.replace_masked_samples(samples, mask)
    assert np.allclose(r, y)
    r, tri = ConvolveBase.replace_masked_samples(samples, mask, get_tri=True)
    assert np.allclose(r, y)
    assert tri is None

    mask.fill(False)
    r, tri = ConvolveBase.replace_masked_samples(samples, mask, get_tri=True)
    assert np.isnan(r).all()
    assert tri is None

    mask.fill(True)
    mask[50] = False
    samples[1, 50] = np.nan
    r, tri = ConvolveBase.replace_masked_samples(samples, mask, get_tri=True)
    assert np.allclose(r, y)
    assert np.allclose(np.delete(x, 50), tri)

    samples, mask = data2d
    r, tri = ConvolveBase.replace_masked_samples(samples, mask=mask,
                                                 get_tri=True)
    assert np.allclose(r, 2)
    assert isinstance(tri, Delaunay)


def test_replace_masked_error(data2d):
    samples, mask = data2d
    error = np.full(mask.size, 2.0)
    error[~mask] = 0
    s = ConvolveBase(*samples, None, mask=mask, error=error)
    _, s._tri = ConvolveBase.replace_masked_samples(samples, mask=mask,
                                                    get_tri=True)
    s.do_error = False
    s.replace_masked_error()
    assert s._interpolated_error is None

    s.do_error = True
    m = mask.copy()
    s.mask.fill(True)
    s.replace_masked_error()
    assert s._interpolated_error is None

    s.mask = m
    s.replace_masked_error()
    assert np.allclose(s._interpolated_error[~m], np.sqrt(14))
    assert np.allclose(s._interpolated_error[m], 2)


def test_replace_1d_masked_error():

    x = np.arange(50)
    y = np.full(50, 2.0)
    mask = np.full(50, True)
    y[25] = np.nan
    mask[25] = False
    samples = stack(x, y)
    error = np.full(50, 2.0)
    error[25] = 0

    s = ConvolveBase(*samples, None, mask=mask, error=error)
    _, s._tri = ConvolveBase.replace_masked_samples(samples, mask=mask,
                                                    get_tri=True)
    s.do_error = False
    s.replace_masked_error()
    assert s._interpolated_error is None

    s.do_error = True
    m = mask.copy()
    s.mask.fill(True)
    s.replace_masked_error()
    assert s._interpolated_error is None

    s.mask = m
    s.replace_masked_error()
    assert np.allclose(s._interpolated_error[~m], np.sqrt(6))
    assert np.allclose(s._interpolated_error[m], 2)


def test_convolve(data2d):
    samples, mask = data2d
    s = ConvolveBase(*samples, None)
    s._convolve(np.zeros((3, 3)))
    assert s._result.shape == (9,)


def test_initial_fit(data2d):
    samples, mask = data2d
    s = ConvolveBase(*samples, None)
    s.initial_fit()
    assert s.covariance is None
    assert np.allclose(s._interpolated_error, 1)


def test_refit_mask(data2d):
    samples, mask = data2d
    s = ConvolveBase(*samples, None)
    s.refit_mask(mask)
    assert s.success
    assert s.stats.dof == 2499

    s.refit_mask(np.full_like(mask, False))
    assert not s.success
    assert s.stats.dof == 0


def test_evaluate(data2d):
    samples, mask = data2d
    s = ConvolveBase(*samples, None)
    r = s.evaluate(None, dovar=False)
    assert r.shape == (2500,)
    r = s.evaluate(None, dovar=True)
    assert r[0].shape == (2500,)
    assert r[1] is None


def test_call(data2d):
    samples, mask = data2d
    s = ConvolveBase(*samples, None)
    assert s(1) is None
