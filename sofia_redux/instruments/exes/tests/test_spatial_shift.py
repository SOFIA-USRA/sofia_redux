# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

import numpy as np
import numpy.testing as npt

from sofia_redux.instruments.exes import spatial_shift as ss
from sofia_redux.toolkit.utilities.fits import set_log_level


class TestSpatialShift(object):

    @pytest.mark.parametrize('shift,shift_str,actual_shift',
                             [(0, '[0 0]', 0),
                              (1, '[ 0 -1]', 1),
                              (-1, '[0 1]', 1),
                              (2, '[ 1 -1]', 2),
                              (4, '[0 0]', 0)])
    def test_spatial_shift(self, capsys, nsb_single_order_hdul,
                           shift, shift_str, actual_shift):
        header = nsb_single_order_hdul[0].header
        data = nsb_single_order_hdul[0].data
        variance = nsb_single_order_hdul[1].data ** 2
        flat = nsb_single_order_hdul[3].data

        if shift == 0:
            sd, sv = ss.spatial_shift(data, header, flat, variance)

            # no shift for synthetic data: traces align
            assert np.allclose(sd, data)
            assert np.allclose(sv, variance)
            assert f'shifts for all frames: ' \
                   f'{shift_str}' in capsys.readouterr().out
        else:
            # shift the second frame: should recover original data,
            # other than top and bottom rows, if average shift < 1
            # (if >= 1, will remove average and shift first frame too)
            shifted = data.copy()
            shifted[1] = ss._shift_2d_array(shifted[1], shift)
            shifted_var = variance.copy()
            shifted_var[1] = ss._shift_2d_array(shifted_var[1], shift)
            with set_log_level('DEBUG'):
                sd, sv = ss.spatial_shift(shifted, header, flat, shifted_var)
            if 0 < actual_shift < 2:
                assert np.allclose(
                    sd[:, actual_shift:-actual_shift - 1, :],
                    data[:, actual_shift:-actual_shift - 1, :])
                assert np.allclose(
                    sv[:, actual_shift:-actual_shift - 1, :],
                    variance[:, actual_shift:-actual_shift - 1, :])
                assert f'shifts for all frames: ' \
                       f'{shift_str}' in capsys.readouterr().out
            else:
                capt = capsys.readouterr()
                assert f'shifts for all frames: {shift_str}' in capt.out
                if actual_shift == 0:
                    assert 'out of range' in capt.out

    def test_verify_inputs(self, rdc_low_hdul):
        header = rdc_low_hdul[0].header
        data = rdc_low_hdul[0].data
        variance = rdc_low_hdul[1].data ** 2
        flat = np.ones(data.shape[1:])
        illum = None
        good_frames = None
        sharpen = True
        params = ss._verify_inputs(
            data, header, flat, variance, illum,
            good_frames, sharpen)

        correct_nz = 4
        assert params['nz'] == correct_nz
        npt.assert_array_equal(params['good_frames'], np.arange(correct_nz))
        npt.assert_array_equal(params['illum'], np.ones_like(flat))

        good_frames = [8, 9, 10]
        with pytest.raises(RuntimeError) as msg:
            ss._verify_inputs(data, header, flat, variance, illum,
                              good_frames, sharpen)
        assert 'No good frames' in str(msg)

        good_frames = [0, 1]
        params = ss._verify_inputs(data, header, flat, variance, illum,
                                   good_frames, sharpen)
        npt.assert_array_equal(params['good_frames'], np.arange(2))

        with pytest.raises(RuntimeError) as err:
            ss._verify_inputs(data[:, :, 0], header, flat, variance, illum,
                              good_frames, sharpen)
        assert 'wrong dimensions' in str(err)

    def test_make_all_templates(self):
        nz, ny, nx = 5, 10, 20
        params = {'header': fits.Header({'NSPAT': nx, 'NSPEC': ny}),
                  'data': np.ones((nz, ny, nx)),
                  'variance': np.ones((nz, ny, nx)),
                  'flat': np.ones((ny, nx)),
                  'illum': np.ones((ny, nx)),
                  'good_frames': []}

        # no good frames
        ss._make_all_templates(params)
        assert len(params['data_templates']) == nz
        assert all([t is None for t in params['data_templates']])
        assert len(params['std_templates']) == nz
        assert all([t is None for t in params['std_templates']])
        assert params['weight_template'].shape == (ny,)
        assert np.allclose(params['weight_template'], 10)

        # some good frames
        params['good_frames'] = [1, 2, 3]
        ss._make_all_templates(params)
        assert len(params['data_templates']) == nz
        assert len(params['std_templates']) == nz
        for i in range(nz):
            template = params['data_templates'][i]
            if i not in [1, 2, 3]:
                assert template is None
            else:
                assert template.shape == (ny,)
                assert np.allclose(template, 1)

    def test_shift_2d(self):
        data = np.arange(16).reshape(4, 4)

        shifted = ss._shift_2d_array(data.copy(), 0)
        assert np.all(shifted == [[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15]])

        shifted = ss._shift_2d_array(data.copy(), 1)
        assert np.all(shifted == [[0, 1, 2, 3],
                                  [0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])

        shifted = ss._shift_2d_array(data.copy(), -1)
        assert np.all(shifted == [[4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15],
                                  [12, 13, 14, 15]])

    def test_shift_1d(self):
        data = np.arange(8)

        shifted = ss._shift_1d_array(data.copy(), 0)
        assert np.all(shifted == [0, 1, 2, 3, 4, 5, 6, 7])

        shifted = ss._shift_1d_array(data.copy(), 1)
        assert np.all(shifted == [0, 0, 1, 2, 3, 4, 5, 6])

        shifted = ss._shift_1d_array(data.copy(), -1)
        assert np.all(shifted == [1, 2, 3, 4, 5, 6, 7, 7])

    def test_find_shift(self):
        nz, ny = 3, 10
        params = {'ns': 100, 'nz': nz,
                  'data_templates': [None, np.full(ny, 4.0), None],
                  'std_templates': [None, np.full(ny, 2.0), None],
                  'weight_template': np.ones(ny),
                  'sharpen': True}
        ss._find_shift(params)
        assert np.all(params['derived_shifts'] == 0)
