# Licensed under a 3-clause BSD style license - see LICENSE.rst

from copy import deepcopy

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.instruments.exes import debounce as ed
from sofia_redux.instruments.exes import readhdr as rd


@pytest.fixture
def basic_params():
    nz, ny, nx = 4, 10, 5
    data = np.ones((nz, ny, nx))
    variance = np.ones((nz, ny, nx))
    mask = np.full(data.shape, True)
    flat = np.ones((ny, nx))
    illum = None
    abeams = [1, 3]
    bbeams = [0, 2]
    spectral = False

    header = fits.Header({'BOUNCE': 0.1, 'NSPAT': nx, 'NSPEC': ny,
                          'INSTMODE': 'NOD_OFF_SLIT', 'INSTCFG': 'LOW'})
    header = rd.readhdr(header, check_header=False)

    params = {'data': data, 'header': header,
              'a_beams': abeams, 'b_beams': bbeams,
              'flat': flat, 'data_mask': mask, 'illum': illum,
              'variance': variance, 'spectral': spectral}
    return params


@pytest.fixture
def derive_params(basic_params):
    param = ed._check_inputs(**basic_params)
    ed._check_nonzero_data(param)
    ed._check_neighbors(param)
    ed._check_nonzero_data(param)
    return param


class TestDebounce(object):

    def single_order_params(self, rdc_low_hdul, spectral=False):
        data = rdc_low_hdul[0].data
        variance = rdc_low_hdul[1].data ** 2
        header = rdc_low_hdul[0].header
        abeams = [1, 3]
        bbeams = [0, 2]
        flat = np.ones(data.shape[1:])
        illum = np.ones(data.shape[1:])
        mask = np.full(data.shape, True)

        header['BOUNCE'] = -0.1

        param = {'data': data, 'header': header,
                 'a_beams': abeams, 'b_beams': bbeams,
                 'flat': flat, 'data_mask': mask, 'illum': illum,
                 'variance': variance, 'spectral': spectral}

        param = ed._check_inputs(**param)
        ed._check_nonzero_data(param)
        ed._check_neighbors(param)
        ed._check_nonzero_data(param)
        return param

    def test_check_inputs(self, basic_params):
        param = ed._check_inputs(**basic_params)

        pass_along = ['data', 'variance', 'header', 'flat',
                      'a_beams', 'b_beams', 'spectral']
        for key in pass_along:
            assert param[key] is basic_params[key]

        assert param['data_mask'].shape == param['data'].shape[1:]
        assert np.all(param['data_mask'])
        assert param['frame_mask'].shape == (param['data'].shape[0],)
        assert np.all(param['frame_mask'])
        assert param['illum'].shape == param['data'].shape[1:]
        assert np.all(param['illum'] == 1)

        assert param['modes'] == {'scan': False, 'nod': True,
                                  'crossdisp': False}
        assert param['do_var'] is True

        assert param['nz'] == param['data'].shape[0]
        assert param['ny'] == param['header']['NSPEC']
        assert param['nx'] == param['header']['NSPAT']
        assert param['bounce'] == param['header']['BOUNCE']

    def test_check_input_errors(self, basic_params):
        # missing bounce factor
        basic_params['header']['BOUNCE'] = 0
        with pytest.raises(ValueError) as err:
            ed._check_inputs(**basic_params)
        assert 'Bounce factor = 0' in str(err)
        basic_params['header']['BOUNCE'] = 0.1

        # bad data shapes raise error
        data = basic_params['data']
        basic_params['data'] = np.zeros(5)
        with pytest.raises(ValueError) as err:
            ed._check_inputs(**basic_params)
        assert 'Data has wrong dimensions' in str(err)
        basic_params['data'] = np.zeros((5, 5, 5, 5))
        with pytest.raises(ValueError) as err:
            ed._check_inputs(**basic_params)
        assert 'Data has wrong dimensions' in str(err)
        basic_params['data'] = data

        # bad var shapes raise error
        var = basic_params['variance']
        basic_params['variance'] = np.zeros(5)
        with pytest.raises(ValueError) as err:
            ed._check_inputs(**basic_params)
        assert 'Variance has wrong dimensions' in str(err)
        basic_params['variance'] = np.zeros((5, 5, 5, 5))
        with pytest.raises(ValueError) as err:
            ed._check_inputs(**basic_params)
        assert 'Variance has wrong dimensions' in str(err)

        # none is okay for variance
        basic_params['variance'] = None
        param = ed._check_inputs(**basic_params)
        assert param['variance'] is None
        assert param['do_var'] is False

        basic_params['variance'] = var

        # beams must match
        a = basic_params['a_beams']
        basic_params['a_beams'] = [1]
        with pytest.raises(ValueError) as err:
            ed._check_inputs(**basic_params)
        assert 'A and B beams must be specified' in str(err)
        basic_params['a_beams'] = a

        # input data mask has to match data
        mask = basic_params['data_mask']
        basic_params['data_mask'] = np.zeros(5)
        with pytest.raises(ValueError) as err:
            ed._check_inputs(**basic_params)
        assert 'mask has wrong dimensions' in str(err)
        basic_params['data_mask'] = mask

    @pytest.mark.parametrize('instmode,instcfg,scan,nod,crossdisp',
                             [('map', 'high_med', True, False, True),
                              ('filemap', 'high_low', True, False, True),
                              ('nod_on_slit', 'medium', False, True, False),
                              ('nod_off_slit', 'low', False, True, False),
                              ('bogus', 'null', False, False, False)])
    def test_check_obsmode(self, instmode, instcfg, scan, nod, crossdisp):
        header = fits.Header()
        header['INSTMODE'] = instmode
        header['INSTCFG'] = instcfg

        result = ed._check_obsmode(header)

        assert isinstance(result, dict)
        assert result['scan'] is scan
        assert result['nod'] is nod
        assert result['crossdisp'] is crossdisp

    def test_check_masks(self):
        nx = 5
        ny = 10
        nz = 4
        data = np.zeros((nz, ny, nx))

        data_mask, frame_mask = ed._check_masks(None, data)
        assert data_mask.shape == (ny, nx)
        assert data_mask.dtype == np.bool
        assert frame_mask.shape == (nz,)
        assert frame_mask.dtype == np.bool
        assert np.all(data_mask)
        assert np.all(frame_mask)

        in_mask = np.array([[[True, True], [True, False]],
                            [[False, True], [True, False]],
                            [[False, False], [False, False]]])

        # error if input mask does not match data
        with pytest.raises(RuntimeError):
            ed._check_masks(in_mask, data)

        data = np.zeros(in_mask.shape)
        data_mask, frame_mask = ed._check_masks(in_mask, data)
        assert data_mask.dtype == np.bool
        assert frame_mask.dtype == np.bool
        assert data_mask.shape == in_mask.shape[-2:]
        assert not np.all(data_mask)
        assert data_mask.sum() == 2
        assert frame_mask.sum() == 2

    @pytest.mark.parametrize('a,b,nz,fail',
                             [([1, 2], [3, 4], 4, False),
                              ([1, 2], [3, 4], 2, True),
                              ([], [3, 4], 4, True),
                              ([1, 2], [], 4, True),
                              ([1, 2], [4], 4, True),
                              ([1, 2], [3, 4], 5, False)])
    def test_check_beams(self, a, b, nz, fail):
        if fail:
            with pytest.raises(RuntimeError):
                ed._check_beams(a, b, nz)
        else:
            ed._check_beams(a, b, nz)

    def test_bounce_confusion(self):
        header = fits.Header()
        header['NODAMP'] = 2
        header['PLTSCALE'] = 4
        header['SPACING'] = 3
        header['NBELOW'] = 2

        params = dict()
        params['header'] = header
        params['modes'] = {'nod': False}

        result = ed._bounce_confusion(params)
        assert result is False

        params['modes']['nod'] = True
        result = ed._bounce_confusion(params)
        assert result is False

        params['header']['NODAMP'] = 4
        params['header']['PLTSCALE'] = 5
        result = ed._bounce_confusion(params)
        assert result is True

        params['header']['NODAMP'] = -9999
        result = ed._bounce_confusion(params)
        assert result is False

    def test_check_nonzero_data(self):
        nz, ny, nx = 4, 5, 6
        params = {'data': np.ones((nz, ny, nx)),
                  'frame_mask': np.full(nz, True)}
        ed._check_nonzero_data(params)
        assert params['data_nonzero'].shape == (ny, nx)
        assert np.all(params['data_nonzero'])

        # all zero in one frame means no good pixels
        params['data'][0] = 0
        ed._check_nonzero_data(params)
        assert np.all(~params['data_nonzero'])

        # unless marked as a bad frame
        params['frame_mask'][0] = False
        ed._check_nonzero_data(params)
        assert np.all(params['data_nonzero'])

        # otherwise, pixels bad in any frame are marked
        params['data'][1, 2, 3] = 0
        params['data'][2, 3, 4] = np.nan
        ed._check_nonzero_data(params)
        assert not np.all(params['data_nonzero'])
        assert params['data_nonzero'].sum() == nx * ny - 2

    def test_check_neighbor_bounds(self, basic_params):
        param = ed._check_inputs(**basic_params)
        ed._check_nonzero_data(param)
        assert 'ok_idx' not in param

        # data good, not spectral, not cross-dispersed: bounds in x only
        ed._check_neighbors(param)
        assert param['direction'] == 'y'
        assert param['ok_idx'].sum() == (param['nx'] - 2) * param['ny']

        # spectral, not cross-dispersed: bounds in y
        param['spectral'] = True
        ed._check_neighbors(param)
        assert param['direction'] == 'x'
        assert param['ok_idx'].sum() == param['nx'] * (param['ny'] - 2)

        # spectral, cross-dispersed: bounds in x
        param['modes']['crossdisp'] = True
        ed._check_neighbors(param)
        assert param['direction'] == 'y'
        assert param['ok_idx'].sum() == (param['nx'] - 2) * param['ny']

        # not spectral, cross-dispersed: bounds in x
        param['spectral'] = False
        ed._check_neighbors(param)
        assert param['direction'] == 'x'
        assert param['ok_idx'].sum() == param['nx'] * (param['ny'] - 2)

    def test_check_neighbor_data(self, basic_params):
        param = ed._check_inputs(**basic_params)
        ed._check_nonzero_data(param)
        assert 'ok_idx' not in param

        # not spectral, not cross-dispersed: bounds in x only
        bounds = (param['nx'] - 2) * param['ny']
        ed._check_neighbors(param)
        assert param['direction'] == 'y'
        assert param['ok_idx'].sum() == bounds

        # one bad point
        param['data'][1, 2, 3] = np.nan
        ed._check_nonzero_data(param)
        ed._check_neighbors(param)
        assert param['ok_idx'].sum() == bounds - 1

        # all bad points
        param['data'][:] = np.nan
        ed._check_nonzero_data(param)
        with pytest.raises(ValueError) as err:
            ed._check_neighbors(param)
        assert 'No good pixels' in str(err)

    def test_shift_2d(self):
        data = np.arange(16).reshape(4, 4)

        shifted = ed._shift_2d_array(data.copy(), 0, 0)
        assert np.all(shifted == [[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15]])

        shifted = ed._shift_2d_array(data.copy(), 1, 0)
        assert np.all(shifted == [[0, 1, 2, 3],
                                  [0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])

        shifted = ed._shift_2d_array(data.copy(), -1, 0)
        assert np.all(shifted == [[4, 5, 6, 7],
                                  [8, 9, 10, 11],
                                  [12, 13, 14, 15],
                                  [12, 13, 14, 15]])

        shifted = ed._shift_2d_array(data.copy(), 1, 1)
        assert np.all(shifted == [[0, 0, 1, 2],
                                  [4, 4, 5, 6],
                                  [8, 8, 9, 10],
                                  [12, 12, 13, 14]])

        shifted = ed._shift_2d_array(data.copy(), -1, 1)
        assert np.all(shifted == [[1, 2, 3, 3],
                                  [5, 6, 7, 7],
                                  [9, 10, 11, 11],
                                  [13, 14, 15, 15]])

    def test_derive_bounce_blank(self, capsys, derive_params):
        param = derive_params

        # no background features
        success = ed._derive_bounce_for_pairs(param)
        assert success is False
        assert "Can't find best 1st derivative" in capsys.readouterr().err

    def test_derive_bounce_feature_large(self, capsys, derive_params):
        param = derive_params

        # add a background feature to all nods,
        # offset for A and B
        param['data'][0, 4, :] += 1
        param['data'][1, 5, :] += 1
        param['data'][2, 4, :] += 1
        param['data'][3, 5, :] += 1

        # add a signal to A
        param['data'][1] += 2
        param['data'][3] += 2

        # bounce amp is too high
        success = ed._derive_bounce_for_pairs(param)
        assert success is False
        assert 'No good frames remaining' in capsys.readouterr().err

    def test_derive_bounce_feature_success(self, capsys, derive_params):
        param = derive_params

        # add a background feature to all nods,
        # offset for A and B
        param['data'][0, 4, :] += 1
        param['data'][1, 5, :] += 1
        param['data'][2, 4, :] += 1
        param['data'][3, 5, :] += 1

        # add a signal to A
        param['data'][1] += 2
        param['data'][3] += 2

        # set the allowed bounce amp higher
        param['bounce'] = 2.0

        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert len(param['first_deriv_shift']) == 2
        assert np.allclose(param['first_deriv_shift'], -1.38724)
        assert np.allclose(param['second_deriv_shift'], 0)

        # mean should be same, but difference between nods
        # should be more even for debounced data
        nsb_nobounce = param['data'][1] - param['data'][0]
        nsb_bounce = param['bounce_data'][1] - param['bounce_data'][0]
        assert np.allclose(np.mean(nsb_bounce), np.mean(nsb_nobounce))
        assert np.std(nsb_bounce) < np.std(nsb_nobounce)

    def test_derive_bounce_feature_2nd_deriv(self, capsys, derive_params):
        param = derive_params

        # add a background feature to all nods,
        # offset for A and B
        param['data'][0, 4, :] += 1
        param['data'][1, 5, :] += 1
        param['data'][2, 4, :] += 1
        param['data'][3, 5, :] += 1

        # add a signal to A
        param['data'][1] += 2
        param['data'][3] += 2

        # set the allowed bounce amp high enough, allow 2nd deriv,
        param['bounce'] = -2.0

        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert len(param['first_deriv_shift']) == 2
        assert np.allclose(param['first_deriv_shift'], -1.38724)
        assert len(param['second_deriv_shift']) == 2
        assert np.allclose(param['second_deriv_shift'], 0)

        # redo, allowing small shifts
        param['skip_small'] = False

        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert len(param['first_deriv_shift']) == 2
        assert np.allclose(param['first_deriv_shift'], -1.38724)
        assert len(param['second_deriv_shift']) == 2
        assert np.allclose(param['second_deriv_shift'], -0.0060955)

        # mean should be same, but difference between nods
        # should be more even for debounced data
        nsb_nobounce = param['data'][1] - param['data'][0]
        nsb_bounce = param['bounce_data'][1] - param['bounce_data'][0]
        assert np.allclose(np.mean(nsb_bounce), np.mean(nsb_nobounce))
        assert np.std(nsb_bounce) < np.std(nsb_nobounce)

    def test_derive_bounce_scan_blank(self, capsys, derive_params):
        param = derive_params

        # set to scan mode
        param['modes']['scan'] = True

        # no background features, but allowed for scan modes
        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert "Can't find best 1st derivative" in capsys.readouterr().err
        assert np.allclose(param['first_deriv_shift'], 0)
        assert np.allclose(param['second_deriv_shift'], 0)

    def test_derive_bounce_scan_large(self, capsys, derive_params):
        param = derive_params

        # set to scan mode
        param['modes']['scan'] = True

        # add a background feature to all nods,
        # offset for A and B
        param['data'][0, 4, :] += 1
        param['data'][1, 5, :] += 1
        param['data'][2, 4, :] += 1
        param['data'][3, 5, :] += 1

        # add a signal to A
        param['data'][1] += 2
        param['data'][3] += 2

        # bounce amp is too high: no error for scan mode
        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert len(param['first_deriv_shift']) == 2
        assert np.allclose(param['first_deriv_shift'], 0)
        assert np.allclose(param['second_deriv_shift'], 0)
        assert param['nzero'] == 2

    def test_derive_bounce_scan_success(self, capsys, derive_params):
        param = derive_params

        # set to scan mode
        param['modes']['scan'] = True

        # add a background feature to all nods,
        # offset for A and B
        param['data'][0, 4, :] += 1
        param['data'][1, 5, :] += 1
        param['data'][2, 4, :] += 1
        param['data'][3, 5, :] += 1

        # add a signal to A
        param['data'][1] += 2
        param['data'][3] += 2

        # set the allowed bounce amp higher
        param['bounce'] = 2.0

        # no recalculation => different shift than for non-scan
        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert len(param['first_deriv_shift']) == 2
        assert np.allclose(param['first_deriv_shift'], -1.973703)
        assert np.allclose(param['second_deriv_shift'], 0)

        # mean should be same, but difference between nods
        # should be more even for debounced data
        nsb_nobounce = param['data'][1] - param['data'][0]
        nsb_bounce = param['bounce_data'][1] - param['bounce_data'][0]
        assert np.allclose(np.mean(nsb_bounce), np.mean(nsb_nobounce))
        assert np.std(nsb_bounce) < np.std(nsb_nobounce)

    def test_derive_bounce_scan_2nd_deriv(self, capsys, derive_params):
        param = derive_params

        # set to scan mode
        param['modes']['scan'] = True

        # add a background feature to all nods,
        # offset for A and B
        param['data'][0, 4, :] += 1
        param['data'][1, 5, :] += 1
        param['data'][2, 4, :] += 1
        param['data'][3, 5, :] += 1

        # add a signal to A
        param['data'][1] += 2
        param['data'][3] += 2

        # set the allowed bounce amp high enough, allow 2nd deriv,
        param['bounce'] = -2.0

        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert len(param['first_deriv_shift']) == 2
        assert np.allclose(param['first_deriv_shift'], -1.973703)
        assert len(param['second_deriv_shift']) == 2
        assert np.allclose(param['second_deriv_shift'], 0)

        # redo, allowing small shifts
        param['skip_small'] = False

        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert len(param['first_deriv_shift']) == 2
        assert np.allclose(param['first_deriv_shift'], -1.973703)
        assert len(param['second_deriv_shift']) == 2
        assert np.allclose(param['second_deriv_shift'], -0.006293)

        # mean should be same, but difference between nods
        # should be more even for debounced data
        nsb_nobounce = param['data'][1] - param['data'][0]
        nsb_bounce = param['bounce_data'][1] - param['bounce_data'][0]
        assert np.allclose(np.mean(nsb_bounce), np.mean(nsb_nobounce))
        assert np.std(nsb_bounce) < np.std(nsb_nobounce)

    def test_derive_bounce_no_frames(self, capsys, derive_params):
        param = derive_params

        param['frame_mask'][:] = False

        # with no good frames, it claims success, with 0 for both shifts
        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert np.allclose(param['first_deriv_shift'], 0)
        assert np.allclose(param['second_deriv_shift'], 0)

    def test_derive_bounce_recalc1(self, mocker, capsys, derive_params):
        mocker.patch.object(ed, '_var_application', return_value=[1, 3])

        param = deepcopy(derive_params)
        success = ed._derive_bounce_for_pairs(param)
        assert not success
        assert "Can't recalculate 1st derivative" in capsys.readouterr().err

        param = deepcopy(derive_params)
        param['modes']['scan'] = True
        success = ed._derive_bounce_for_pairs(param)
        assert success
        assert np.allclose(param['first_deriv_shift'], 0.15)

    def test_derive_bounce_recalc2(self, mocker, capsys, derive_params):

        # patch var application to return something different
        # on the final call
        class RecalcVar(object):
            def __init__(self, final=(-1, 1), bad_call=(3, 6)):
                self.n_call = 0
                self.final = final
                self.bad_call = bad_call

            def __call__(self, var):
                self.n_call += 1
                if self.n_call not in self.bad_call:
                    return 1, 1
                else:
                    return self.final

        # 2nd deriv calculation failure (1st deriv okay)
        mocker.patch.object(ed, '_var_application', RecalcVar())
        param = deepcopy(derive_params)
        param['bounce'] = -0.1
        success = ed._derive_bounce_for_pairs(param)
        assert not success
        assert "Can't find best 2nd derivative" in capsys.readouterr().err

        # same for scan - succeeds, reports 0
        mocker.patch.object(ed, '_var_application', RecalcVar(bad_call=(2, 4)))
        param = deepcopy(derive_params)
        param['bounce'] = -0.1
        param['modes']['scan'] = True
        success = ed._derive_bounce_for_pairs(param)
        assert success
        assert "Can't find best 2nd derivative" in capsys.readouterr().err
        assert np.allclose(param['second_deriv_shift'], 0)

        # 2nd deriv too big
        mocker.patch.object(ed, '_var_application',
                            RecalcVar(final=(1, 9), bad_call=(3, 6)))
        param = deepcopy(derive_params)
        param['bounce'] = -0.1
        success = ed._derive_bounce_for_pairs(param)
        assert not success
        assert "No good frames" in capsys.readouterr().err

        # same for scan
        mocker.patch.object(ed, '_var_application',
                            RecalcVar(final=(1, 9), bad_call=(2, 4)))
        param = deepcopy(derive_params)
        param['bounce'] = -0.1
        param['modes']['scan'] = True
        success = ed._derive_bounce_for_pairs(param)
        assert success
        assert "No good frames" not in capsys.readouterr().err
        assert np.allclose(param['second_deriv_shift'], 0)

        # 2nd deriv okay, not recalculated
        mocker.patch.object(ed, '_var_application',
                            RecalcVar(final=(1, 0.1), bad_call=(2, 4)))
        param = deepcopy(derive_params)
        param['bounce'] = -0.1
        param['modes']['scan'] = True
        success = ed._derive_bounce_for_pairs(param)
        assert success
        assert np.allclose(param['second_deriv_shift'], 0.005)

        # 2nd deriv too big, okay for scan, returns 0
        mocker.patch.object(ed, '_var_application',
                            RecalcVar(final=(1, 1), bad_call=(2, 4)))
        param = deepcopy(derive_params)
        param['bounce'] = -0.1
        param['modes']['scan'] = True
        success = ed._derive_bounce_for_pairs(param)
        assert success
        assert "Can't recalculate 2nd derivative" in capsys.readouterr().err
        assert np.allclose(param['second_deriv_shift'], 0)

        # 2nd deriv too big, error for not scan, returns 0
        mocker.patch.object(ed, '_var_application',
                            RecalcVar(final=(-1, 1), bad_call=(4, 8)))
        param = deepcopy(derive_params)
        param['bounce'] = -0.1
        success = ed._derive_bounce_for_pairs(param)
        assert not success
        assert "Can't recalculate 2nd derivative" in capsys.readouterr().err

    def test_derive_bounce_full_data(self, capsys, rdc_low_hdul):
        param = self.single_order_params(rdc_low_hdul, spectral=False)
        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert np.allclose(param['first_deriv_shift'], 0.1, atol=0.05)
        assert np.allclose(param['second_deriv_shift'], 0.1, atol=0.05)
        assert param['nzero'] == 0
        assert not np.allclose(param['bounce_data'], param['data'])
        assert capsys.readouterr().err == ''

        param = self.single_order_params(rdc_low_hdul, spectral=True)
        success = ed._derive_bounce_for_pairs(param)
        assert success is True
        assert np.allclose(param['first_deriv_shift'], 0.08, atol=0.05)
        assert np.allclose(param['second_deriv_shift'], 0.02, atol=0.05)
        assert param['nzero'] == 0
        assert not np.allclose(param['bounce_data'], param['data'])
        assert capsys.readouterr().err == ''

    def test_debounce(self, mocker, capsys, basic_params):
        bdata = np.arange(10)
        param = ed._check_inputs(**basic_params)
        param['bounce'] = -0.1
        param['first_deriv_shift'] = 0
        param['second_deriv_shift'] = 0
        param['nzero'] = 1
        param['bounce_data'] = bdata
        m1 = mocker.patch.object(ed, '_check_inputs', return_value=param)
        m2 = mocker.patch.object(ed, '_bounce_confusion', return_value=True)
        m3 = mocker.patch.object(ed, '_check_nonzero_data')
        m4 = mocker.patch.object(ed, '_check_neighbors')
        m5 = mocker.patch.object(ed, '_derive_bounce_for_pairs',
                                 return_value=True)

        result = ed.debounce(basic_params['data'], basic_params['header'],
                             basic_params['a_beams'], basic_params['b_beams'],
                             basic_params['flat'])
        assert m1.call_count == 1
        assert m2.call_count == 1
        assert m3.call_count == 2
        assert m4.call_count == 1
        assert m5.call_count == 1
        assert result is bdata

        capt = capsys.readouterr()
        assert 'Nodded source may confuse' in capt.err
        assert 'First derivative' in capt.out
        assert 'Second derivative' in capt.out
        assert 'Setting weight = 0 for 1 pairs' in capt.out

        # set to scan mode
        param['modes']['scan'] = True
        ed.debounce(basic_params['data'], basic_params['header'],
                    basic_params['a_beams'], basic_params['b_beams'],
                    basic_params['flat'])
        assert '1 pairs exceeded bounce limit' not in capt.out

        m6 = mocker.patch.object(ed, '_derive_bounce_for_pairs',
                                 return_value=False)
        result = ed.debounce(basic_params['data'], basic_params['header'],
                             basic_params['a_beams'], basic_params['b_beams'],
                             basic_params['flat'])
        assert m6.call_count == 1
        assert result is basic_params['data']

        m7 = mocker.patch.object(ed, '_check_inputs',
                                 side_effect=ValueError('test'))
        result = ed.debounce(basic_params['data'], basic_params['header'],
                             basic_params['a_beams'], basic_params['b_beams'],
                             basic_params['flat'])
        assert m7.call_count == 1
        assert result is basic_params['data']
        assert 'test' in capsys.readouterr().err
