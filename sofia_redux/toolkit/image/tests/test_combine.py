# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

import sofia_redux.toolkit.image as img
from sofia_redux.toolkit.tests.resources import random_mask, add_jailbars


def fake_data(shape=(256, 256), value=2.0, badfrac=0.1, jblevel=1.0):
    data = np.full(shape, value)
    add_jailbars(data, level=jblevel)
    badmask = random_mask(shape[-2:], frac=badfrac)
    if len(data.shape) == 2:
        data[badmask] = np.nan
    elif len(data.shape) == 3:
        for i, frame in enumerate(data):
            frame += i * 0.1 * value
            frame[badmask] = np.nan
    return data


class TestCombine(object):

    @pytest.mark.parametrize('data', [1, 'string', [1, 2],
                                      [np.zeros((1, 2)), np.zeros((2, 2))],
                                      [np.zeros((2, 2))]])
    def test_data_errors(self, data):
        with pytest.raises(ValueError):
            img.combine.combine_images(data)

    def test_var_errors(self):
        dlist = np.zeros((3, 20, 20))

        # no error if shape matches
        vlist = np.zeros((3, 20, 20))
        img.combine.combine_images(dlist, variance=vlist)

        # errors if it does not
        with pytest.raises(ValueError):
            img.combine.combine_images(dlist, variance=np.zeros((3, 22, 20)))
        with pytest.raises(ValueError):
            img.combine.combine_images(dlist, variance=np.zeros((2, 20, 20)))

    def test_options(self):
        dlist = np.zeros((2, 10, 10))
        expected = np.zeros((10, 10))

        # fails if bad method name provided
        with pytest.raises(ValueError):
            img.combine.combine_images(dlist, method='average')

        # accepts kwargs, ignores axis and masked if provided
        kwargs = {'sigma': 3, 'axis': 1, 'masked': False}
        data, var = img.combine.combine_images(dlist, robust=True, **kwargs)
        assert np.allclose(data, expected)
        assert np.allclose(var, expected)

        # throws error for unrecognized argument
        kwargs['test_arg'] = True
        with pytest.raises(TypeError):
            img.combine.combine_images(dlist, robust=True, **kwargs)

    def test_methods(self):
        dval = 2.0
        vval = 0.1
        dlist = fake_data(shape=(5, 50, 50), value=dval)
        vlist = fake_data(shape=(5, 50, 50), value=vval)
        nanidx = np.all(np.isnan(dlist), axis=0) \
            | np.all(np.isnan(vlist), axis=0)
        nn = ~nanidx

        # test mean methods
        m1, v1 = img.combine.combine_images(dlist, variance=vlist,
                                            method='mean', robust=False,
                                            weighted=False)
        m2, v2 = img.combine.combine_images(dlist, variance=vlist,
                                            method='mean', robust=True,
                                            weighted=False)
        m3, v3 = img.combine.combine_images(dlist, variance=vlist,
                                            method='mean', robust=False,
                                            weighted=True)
        m4, v4 = img.combine.combine_images(dlist, variance=vlist,
                                            method='mean', robust=True,
                                            weighted=True)

        # test median methods
        m5, v5 = img.combine.combine_images(dlist, variance=vlist,
                                            method='median', robust=True)
        m6, v6 = img.combine.combine_images(dlist, variance=vlist,
                                            method='median', robust=False)

        # test without input variance
        m7, v7 = img.combine.combine_images(dlist, method='mean', robust=False)
        m8, v8 = img.combine.combine_images(dlist, method='median',
                                            robust=False)

        # test sum method
        m9, v9 = img.combine.combine_images(dlist, method='sum',
                                            variance=vlist, robust=False)
        m10, v10 = img.combine.combine_images(dlist, method='sum',
                                              robust=False)

        # all mean/median answers should be pretty close

        # robust and not should be same -- data is very even
        assert np.allclose(m1[nn], m2[nn])
        assert np.allclose(m1[nn], m5[nn])
        assert np.allclose(m1[nn], m6[nn])
        assert np.allclose(m1[nn], m7[nn])
        assert np.allclose(m1[nn], m8[nn])

        # weighted should be same within error
        assert np.max(np.abs(m1[nn] - m3[nn])) < np.sqrt(np.max(v1[nn]))
        assert np.max(np.abs(m1[nn] - m4[nn])) < np.sqrt(np.max(v1[nn]))

        # all mean variances should be pretty close
        assert np.allclose(v1[nn], v2[nn], atol=1e-3)
        assert np.allclose(v1[nn], v3[nn], atol=1e-3)
        assert np.allclose(v1[nn], v4[nn], atol=1e-3)

        # median variances are a factor of pi/2 different
        assert np.allclose(v1[nn] * np.pi / 2, v5[nn], atol=1e-3)
        assert np.allclose(v1[nn] * np.pi / 2, v6[nn], atol=1e-3)

        # intrinsic variances when not propagated
        inp_var = np.nanvar(dlist, axis=0)
        assert np.allclose(inp_var, v7, equal_nan=True)
        assert np.allclose(inp_var, v8, equal_nan=True)

        # directly check sums
        direct_sum = np.nansum(dlist, axis=0)
        direct_sum_var = np.nansum(vlist, axis=0)
        assert np.max(np.abs(direct_sum[nn] - m9[nn])) == 0
        assert np.max(np.abs(direct_sum[nn] - m10[nn])) == 0
        assert np.max(np.abs(direct_sum_var[nn] - v9[nn])) == 0
        assert np.allclose(inp_var[nn], v10[nn])

    def test_not_returned(self):
        dlist = np.zeros((2, 10, 10))
        expected = np.zeros((10, 10))
        m1 = img.combine.combine_images(dlist, method='mean', returned=False)
        m2 = img.combine.combine_images(dlist, method='median', returned=False)
        m3 = img.combine.combine_images(dlist, method='sum', returned=False)
        assert np.allclose(m1, expected)
        assert np.allclose(m2, expected)
        assert np.allclose(m3, expected)

    @pytest.mark.parametrize('method,weighted',
                             [('mean', True), ('mean', False),
                              ('median', False), ('sum', False)])
    def test_nan_variance(self, method, weighted):
        # test that NaNs in variance plane but not in image plane are
        # handled properly
        dval = 2.0
        vval = 0.1
        dlist = fake_data(shape=(5, 50, 50), value=dval)
        vlist = fake_data(shape=(5, 50, 50), value=vval)

        # make sure at least one value is Nan in the whole stack
        dlist[:, 10, 10] = np.nan
        vlist[:, 20, 20] = np.nan

        # check that there are places where the NaNs are different
        assert not np.all(np.isnan(dlist) & np.isnan(vlist))

        # combine data
        m, v = img.combine.combine_images(dlist, variance=vlist, method=method,
                                          weighted=weighted, returned=True)

        # check that combined data is only NaN where all input in
        # either data or variance is Nan
        dnan = np.all(np.isnan(dlist), axis=0) \
            | np.all(np.isnan(vlist), axis=0)
        assert np.all(np.isnan(m[dnan]))
        assert np.all(~np.isnan(m[~dnan]))

        # same for variance
        assert np.all(np.isnan(v[dnan]))
        assert np.all(~np.isnan(v[~dnan]))
