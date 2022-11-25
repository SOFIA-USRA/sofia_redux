# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

import sofia_redux.instruments.exes.tort as et
from sofia_redux.instruments.exes.readhdr import readhdr


class TestTort(object):

    @pytest.mark.parametrize('method,nnan',
                             [('spline', 20), ('convolution', 30)])
    def test_tort(self, method, nnan):
        nz, ny, nx = 5, 10, 20
        data = np.ones((nz, ny, nx))
        variance = np.zeros((nz, ny, nx))
        header = readhdr(fits.Header({'NSPAT': nx, 'NSPEC': ny}),
                         check_header=False)

        # bad data
        with pytest.raises(ValueError) as err:
            et.tort(data[:, 0, :], header, interpolation_method=method)
        assert 'Data cannot be corrected' in str(err)

        # 3D data
        td = et.tort(data, header, interpolation_method=method)
        assert td.shape == (nz, ny, nx)
        assert np.allclose(td[~np.isnan(td)], 1)
        assert np.allclose(td, td[0], equal_nan=True)
        assert np.isnan(td).sum() == nnan * 5

        # 2D data
        td = et.tort(data[0], header, interpolation_method=method)
        assert td.shape == (ny, nx)
        assert np.allclose(td[~np.isnan(td)], 1)
        assert np.isnan(td).sum() == nnan

        # data + variance
        td, tv = et.tort(data, header, variance=variance,
                         interpolation_method=method)
        assert td.shape == (nz, ny, nx)
        assert tv.shape == (nz, ny, nx)
        assert np.allclose(tv[~np.isnan(tv)], 0)
        assert np.allclose(tv, tv[0], equal_nan=True)
        assert np.isnan(tv).sum() == nnan * 5

        td, tv = et.tort(data[0], header, variance=variance[0],
                         interpolation_method=method)
        assert td.shape == (ny, nx)
        assert tv.shape == (ny, nx)
        assert np.allclose(tv[~np.isnan(tv)], 0)
        assert np.isnan(tv).sum() == nnan

        # data + variance + illum
        td, tv, ti = et.tort(data, header, variance=variance, get_illum=True,
                             interpolation_method=method)
        assert td.shape == (nz, ny, nx)
        assert tv.shape == (nz, ny, nx)
        assert ti.shape == (nz, ny, nx)
        assert np.allclose(ti[ti != -1], 1)
        assert np.allclose(ti, ti[0])
        assert (ti == -1).sum() == nnan * 5

        # data + illum
        td, ti = et.tort(data[0], header, get_illum=True,
                         interpolation_method=method)
        assert td.shape == (ny, nx)
        assert ti.shape == (ny, nx)
        assert np.allclose(ti[ti != -1], 1)
        assert (ti == -1).sum() == nnan

    @pytest.mark.parametrize('method', ['spline', 'convolution'])
    def test_tort_full_data(self, single_order_flat, method):
        data = single_order_flat[0].data
        header = single_order_flat[0].header
        variance = single_order_flat[1].data ** 2

        td, tv, ti = et.tort(data, header, variance=variance, get_illum=True,
                             interpolation_method=method)
        assert td.shape == data.shape
        assert tv.shape == data.shape
        assert ti.shape == data.shape
        assert np.allclose(np.nanmean(td), np.mean(data), rtol=.01)

        # missing pixels are unilluminated only
        assert np.all(np.isnan(td) == (ti == -1))
        nansum = np.isnan(td).sum()

        # add a few NaNs
        rand = np.random.RandomState(42)
        idx = rand.random_sample(int(data.size * .005)) * data.size
        data.flat[idx.astype(int)] = np.nan
        td2, tv2, ti2 = et.tort(data, header, variance=variance,
                                get_illum=True,
                                interpolation_method=method)

        # mean is still same
        assert np.allclose(np.nanmean(td2), np.nanmean(td), rtol=.01)
        # more NaNs in output
        nan2 = np.isnan(td2)
        assert nan2.sum() > nansum
        assert not np.all(nan2)

        # unilluminated has not changed,
        # NaN pixels are not unilluminated only
        assert nansum == (ti2 == -1).sum()

    def test_check_data(self, capsys):
        nz, ny, nx = 5, 10, 20
        data = np.ones((nz, ny, nx))
        variance = np.zeros((nz, ny, nx))
        header = fits.Header({'NSPAT': nx, 'NSPEC': ny})

        check, d, v, i = et._check_data(data, header, variance)
        assert check is True
        assert d.shape == (nz, ny, nx)
        assert v.shape == (nz, ny, nx)
        assert i.shape == (nz, ny, nx)
        assert np.all(i == 1)

        check, d, v, i = et._check_data(data[0], header, variance)
        assert check is False
        assert d.shape == (1, ny, nx)
        assert v.shape == (nz, ny, nx)
        assert 'Variance and data shape mismatch' in capsys.readouterr().err

        check, d, v, i = et._check_data(data[0], header, variance[0])
        assert check is True
        assert d.shape == (1, ny, nx)
        assert v.shape == (1, ny, nx)

        check, d, v, i = et._check_data(data[:, 0, :], header, variance)
        assert check is False
        assert d.shape == (1, nz, nx)
        assert v.shape == (nz, ny, nx)
        assert 'Data (y, x) dimensions do not match' in capsys.readouterr().err
