# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import numpy.testing as npt
import pytest

from sofia_redux.instruments.exes import diff_arr as da


class TestDiffArr(object):

    def test_check_beams(self, rdc_header):
        nx = 2
        ny = 3
        nz = 4
        abeams = np.array([0, 2])
        bbeams = np.array([1, 3])
        data = np.array([a + i for i, a in
                         enumerate([np.ones((ny, nx))] * nz)])
        var = data.copy() / 4
        mask = np.zeros_like(data)
        header = rdc_header.copy()
        header['INSTMODE'] = 'MAP'
        info = da._check_beams(abeams, bbeams, header,
                               data, var, True, 4, False, mask)
        assert info['data'].shape == (nz // 2, ny, nx)
        npt.assert_array_equal(info['data'] % 2, 0)
        assert info['var'].shape == (nz // 2, ny, nx)
        npt.assert_equal(info['data_avg'], 3)
        npt.assert_equal(info['var_avg'], 3 / 4)

        # raises error for empty beams
        with pytest.raises(RuntimeError):
            da._check_beams([], bbeams, header, data, var,
                            True, 4, False, mask)

        # does not raise error for mismatched beams in map mode
        # or with black_dark
        da._check_beams([0], bbeams, header, data, var, True, 4, False, mask)
        da._check_beams([0], bbeams, header, data, var, True, 4, True, mask)

        # but does for any other
        header['INSTMODE'] = 'NOD_OFF_SLIT'
        da._check_beams([0], bbeams, header, data, var, True, 4, True, mask)
        with pytest.raises(RuntimeError):
            da._check_beams([0], bbeams, header, data, var, True, 4,
                            False, mask)

    def test_apply_beams_map(self, rdc_header):
        nx = 2
        ny = 3
        nz = 6
        abeams = np.array([1, 2])
        bbeams = np.array([0, 3, 4, 5])
        data = np.array([a + i for i, a in
                         enumerate([np.ones((ny, nx))] * nz)])
        var = data.copy() / 4
        do_var = True
        mask = np.zeros_like(data, dtype=int)

        header = rdc_header.copy()
        header['INSTMODE'] = 'MAP'
        kwargs = {'data': data, 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nz': nz, 'black_dark': False, 'mask': mask}
        info = da._check_beams(**kwargs)

        kwargs = {'data': data.copy(), 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nx': nx, 'ny': ny, 'black_dark': False, 'dark': None,
                  'b_info': info, 'mask': mask}
        dd, dv, dm = da._apply_beams(**kwargs)
        assert dd.shape == (len(abeams), ny, nx)
        assert dv.shape == (len(abeams), ny, nx)
        assert dm.shape == (len(abeams), ny, nx)
        assert np.allclose(dd[0], data[1] - info['data_avg'])
        assert np.allclose(dd[1], data[2] - info['data_avg'])
        assert np.allclose(dv[0], var[1] + info['var_avg'])
        assert np.allclose(dv[1], var[2] + info['var_avg'])
        assert np.all(dm == 0)

    def test_apply_beams_nod(self, rdc_header):
        nx = 2
        ny = 3
        nz = 4
        abeams = np.array([1, 3])
        bbeams = np.array([0, 2])
        data = np.array([a + i for i, a in
                         enumerate([np.ones((ny, nx))] * nz)])
        var = data.copy() / 4
        do_var = True

        header = rdc_header.copy()
        kwargs = {'data': data, 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nz': nz, 'black_dark': False, 'mask': None}
        info = da._check_beams(**kwargs)

        kwargs = {'data': data.copy(), 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nx': nx, 'ny': ny, 'black_dark': False, 'dark': None,
                  'b_info': info, 'mask': None}
        dd, dv, dm = da._apply_beams(**kwargs)
        assert dd.shape == (nz // 2, ny, nx)
        assert dv.shape == (nz // 2, ny, nx)
        assert dm.shape == (nz // 2, ny, nx)
        assert np.allclose(dd[0], data[1] - data[0])
        assert np.allclose(dd[1], data[3] - data[2])
        assert np.allclose(dv[0], var[1] + var[0])
        assert np.allclose(dv[1], var[3] + var[2])
        assert np.all(dm == 0)

    def test_apply_beams_map_dark(self, rdc_header):
        nx = 2
        ny = 3
        nz = 6
        abeams = np.array([1, 2])
        bbeams = np.array([0, 3, 4, 5])
        data = np.array([a + i for i, a in
                         enumerate([np.ones((ny, nx))] * nz)])
        var = data.copy() / 4
        do_var = True
        dark = np.full((ny, nx), 10.0)

        header = rdc_header.copy()
        header['INSTMODE'] = 'MAP'
        kwargs = {'data': data, 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nz': nz, 'black_dark': True, 'mask': None}
        info = da._check_beams(**kwargs)

        kwargs = {'data': data.copy(), 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nx': nx, 'ny': ny, 'black_dark': True, 'dark': dark,
                  'b_info': info, 'mask': None}

        dd, dv, dm = da._apply_beams(**kwargs)
        assert dd.shape == (len(bbeams), ny, nx)
        assert dv.shape == (len(bbeams), ny, nx)
        assert dm.shape == (len(bbeams), ny, nx)
        assert np.allclose(dd[0], data[0] - dark)
        assert np.allclose(dd[1], data[3] - dark)
        assert np.allclose(dd[2], data[4] - dark)
        assert np.allclose(dd[3], data[5] - dark)

        assert np.allclose(dv[0], var[0])
        assert np.allclose(dv[1], var[3])
        assert np.allclose(dv[2], var[4])
        assert np.allclose(dv[3], var[5])
        assert np.all(dm == 0)

    def test_apply_beams_nod_dark(self, rdc_header):
        nx = 2
        ny = 3
        nz = 4
        abeams = np.array([1, 3])
        bbeams = np.array([0, 2])
        data = np.array([a + i for i, a in
                         enumerate([np.ones((ny, nx))] * nz)])
        var = data.copy() / 4
        do_var = True
        dark = np.full((ny, nx), 10.0)

        header = rdc_header.copy()
        kwargs = {'data': data, 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nz': nz, 'black_dark': True, 'mask': None}
        info = da._check_beams(**kwargs)

        kwargs = {'data': data.copy(), 'abeams': abeams, 'bbeams': bbeams,
                  'header': header, 'variance': var, 'do_var': do_var,
                  'nx': nx, 'ny': ny, 'black_dark': True, 'dark': dark,
                  'b_info': info, 'mask': None}
        dd, dv, dm = da._apply_beams(**kwargs)
        assert dd.shape == (nz // 2, ny, nx)
        assert dv.shape == (nz // 2, ny, nx)
        assert dm.shape == (nz // 2, ny, nx)
        assert np.allclose(dd[0], data[0] - dark)
        assert np.allclose(dd[1], data[2] - dark)
        assert np.allclose(dv[0], var[0])
        assert np.allclose(dv[1], var[2])
        assert np.all(dm == 0)

    def test_replace_small_values(self, rdc_header):
        header = rdc_header.copy()
        header['PAGAIN'] = 2
        header['BEAMTIME'] = 2
        diff_data = np.array([[4, 0, 2], [3, 0, 0]]).astype(float)
        diff_var = np.array([[4, 0, 2], [3, 0, 0]]).astype(float)
        kwargs = {'diff_data': diff_data, 'diff_var': diff_var,
                  'header': header, 'do_var': True}
        assert np.count_nonzero(diff_data == 0) == 3
        assert np.count_nonzero(diff_var == 0) == 3
        dd, dv = da._replace_small_values(**kwargs)
        assert np.count_nonzero(dd == 0) == 0
        assert np.count_nonzero(dv == 0) == 0
        assert np.isclose(np.nanmin(dd), 0.25)
        assert np.isclose(np.nanmin(dv), 1)

    def test_diff_arr_full_data(self, capsys, rdc_low_hdul):
        data = rdc_low_hdul[0].data
        variance = rdc_low_hdul[1].data ** 2
        header = rdc_low_hdul[0].header
        abeams = [1, 3]
        bbeams = [0, 2]
        nz, ny, nx = data.shape
        dark = np.ones((ny, nx), dtype=float)

        # add some bad pixels
        mask = np.zeros_like(data).astype(int)
        rand = np.random.RandomState(42)
        idx = rand.random_sample(mask.size // 10) * mask.size
        mask.flat[idx.astype(int)] = 1

        d, v, m = da.diff_arr(data, header, abeams, bbeams, variance,
                              mask=mask)
        assert d.shape == (nz // 2, ny, nx)
        assert v.shape == (nz // 2, ny, nx)
        assert m.shape == (nz // 2, ny, nx)
        assert np.allclose(d[0], data[1] - data[0])
        assert np.allclose(d[1], data[3] - data[2])
        assert np.allclose(m[0], np.any([mask[1], mask[0]],
                                        axis=0).astype(int))
        assert np.allclose(m[1], np.any([mask[3], mask[2]],
                                        axis=0).astype(int))
        # variance values replaced with min from header
        assert np.allclose(v, 4.0)

        # small minimum value
        header['PAGAIN'] = 1e6
        d, v, m = da.diff_arr(data, header, abeams, bbeams, variance, mask)
        assert d.shape == (nz // 2, ny, nx)
        assert v.shape == (nz // 2, ny, nx)
        assert np.allclose(d[0], data[1] - data[0])
        assert np.allclose(d[1], data[3] - data[2])
        assert np.allclose(m[0], np.any([mask[1], mask[0]],
                                        axis=0).astype(int))
        assert np.allclose(m[1], np.any([mask[3], mask[2]],
                                        axis=0).astype(int))
        assert np.allclose(v[0], variance[1] + variance[0])
        assert np.allclose(v[1], variance[3] + variance[2])

        # subtract dark instead
        d, v, m = da.diff_arr(data, header, abeams, bbeams, variance, mask,
                              dark=dark, black_dark=True)
        assert d.shape == (nz // 2, ny, nx)
        assert v.shape == (nz // 2, ny, nx)
        assert m.shape == (nz // 2, ny, nx)
        assert np.allclose(d[0], data[0] - dark)
        assert np.allclose(d[1], data[2] - dark)
        assert np.allclose(m[0], mask[0])
        assert np.allclose(m[1], mask[2])
        assert np.allclose(v[0], variance[0])
        assert np.allclose(v[1], variance[2])

        # errors for mismatched dimensions or beams
        # return data without diffing
        d, v, m = da.diff_arr(data[0, 0], header, abeams, bbeams, variance)
        assert d.shape == (nx,)
        assert v.shape == (nz, ny, nx)
        assert m is None
        assert 'Data has wrong dimensions' in capsys.readouterr().err

        d, v, m = da.diff_arr(data, header, abeams, bbeams, variance[0])
        assert d.shape == (nz, ny, nx)
        assert v.shape == (ny, nx)
        assert m is None
        assert 'Variance has wrong dimensions' in capsys.readouterr().err

        d, v, m = da.diff_arr(data, header, [], bbeams, variance)
        assert d.shape == (nz, ny, nx)
        assert v.shape == (nz, ny, nx)
        assert m is None
        assert 'A and B beams must be specified' in capsys.readouterr().err
