# Licensed under a 3-clause BSD style license - see LICENSE.rst

import logging

from astropy.io import fits
import numpy as np
import numpy.testing as npt
import pytest

from sofia_redux.instruments.exes import coadd, utils, make_template


@pytest.fixture
def basic_params():
    nspec = 10
    nspat = 5
    nz = 5
    data = np.ones((nz, nspec, nspat))
    variance = np.ones((nz, nspec, nspat))
    flat = np.ones((nspec, nspat))
    illum = np.ones((nspec, nspat))
    good_frames = np.array([1, 2, 3, 4])
    weight_mode = 'useweights'
    weights = np.array([0.2, 0.4, 0, 0.2, 0.2])
    std_wt = True
    threshold = 2.5
    header = fits.Header()
    header['NSPEC'] = nspec
    header['NSPAT'] = nspat
    header['INSTCFG'] = 'HIGH_MED'
    params = {'data': data, 'header': header, 'weights': weights,
              'variance': variance, 'flat': flat, 'illum': illum,
              'good_frames': good_frames, 'weight_mode': weight_mode,
              'std_wt': std_wt, 'threshold': threshold, 'nz': nz}
    return params


class TestCoadd(object):

    def test_verify_inputs(self, mocker, basic_params):
        mocker.patch.object(utils, 'check_data_dimensions',
                            return_value=basic_params['nz'])

        del basic_params['nz']
        output = coadd._verify_inputs(**basic_params)

        assert isinstance(output, dict)
        npt.assert_array_equal(output['data'], basic_params['data'])
        npt.assert_array_equal(output['variance'], basic_params['variance'])
        npt.assert_array_equal(output['illum'], basic_params['illum'])
        npt.assert_array_equal(output['flat'], basic_params['flat'])
        assert output['header'] == basic_params['header']
        assert output['nx'] == basic_params['header']['NSPAT']
        assert output['ny'] == basic_params['header']['NSPEC']
        for key in ['weights', 'good_frames']:
            npt.assert_array_equal(output[key], basic_params[key])
        assert output['weight_mode'] == basic_params['weight_mode']
        assert output['std_wt'] == basic_params['std_wt']
        assert output['threshold'] == basic_params['threshold']
        npt.assert_array_equal(output['suball'], [1, 2, 3, 4])
        assert output['zwt_sum'] == 1

    def test_input_errors(self, basic_params):
        del basic_params['nz']

        data = basic_params['data']
        basic_params['data'] = np.ones(1)
        with pytest.raises(ValueError) as err:
            coadd._verify_inputs(**basic_params)
        assert 'Data has wrong dimensions' in str(err)

        var = basic_params['variance']
        basic_params['data'] = data
        basic_params['variance'] = np.ones(1)
        with pytest.raises(ValueError) as err:
            coadd._verify_inputs(**basic_params)
        assert 'Variance has wrong dimensions' in str(err)

        # missing illum is okay
        basic_params['variance'] = var
        basic_params['illum'] = None
        param = coadd._verify_inputs(**basic_params)
        assert param['illum'].shape == data.shape[1:]
        assert np.all(param['illum'] == 1)

    def test_verify_frames(self, basic_params):
        nz = basic_params.pop('nz')
        keys = ['suball', 'zwt_sum']
        for key in keys:
            assert key not in basic_params

        param = coadd._verify_inputs(**basic_params)
        assert np.allclose(param['suball'], np.array([1, 2, 3, 4]))
        assert param['zwt_sum'] == 1

        basic_params['good_frames'] = None
        param = coadd._verify_inputs(**basic_params)
        assert np.allclose(param['suball'], np.array([0, 1, 2, 3, 4]))
        assert param['zwt_sum'] == 1

        basic_params['good_frames'] = []
        with pytest.raises(ValueError) as err:
            coadd._verify_inputs(**basic_params)
        assert 'No good frames' in str(err)

        basic_params['good_frames'] = None
        basic_params['weights'] = None
        param = coadd._verify_inputs(**basic_params)
        assert np.allclose(param['suball'], np.array([0, 1, 2, 3, 4]))
        assert param['zwt_sum'] == 5

        basic_params['weights'] = np.zeros(nz)
        with pytest.raises(ValueError) as err:
            coadd._verify_inputs(**basic_params)
        assert 'All weights are zero' in str(err)

    @pytest.mark.parametrize('mode,unweighted,do_weights,zwt_sum',
                             [('unweighted', True, True, 1),
                              ('useweights', False, False, 1)])
    def test_determine_weighting(self, basic_params, mode, unweighted,
                                 do_weights, zwt_sum):
        basic_params['weight_mode'] = mode

        for key in ['unweighted', 'do_weights', 'zwt_sum']:
            assert key not in basic_params

        coadd._determine_weighting_method(basic_params)

        assert basic_params['unweighted'] == unweighted
        assert basic_params['do_weights'] == do_weights
        assert basic_params['zwt_sum'] == zwt_sum

    def test_determine_weights_equal(self, basic_params):
        basic_params['weight_mode'] = 'unweighted'
        basic_params['template'] = np.ones_like(basic_params['data'][0])
        basic_params['weight_frame'] = np.ones_like(basic_params['data'])
        coadd._determine_weighting_method(basic_params)

        for key in ['sum_wt', 'sum_wt_sq', 'wt_max']:
            assert key not in basic_params

        orig_weights = basic_params['weights'].copy()
        coadd._calculate_weights(basic_params)

        assert basic_params['sum_wt'] == 4
        assert basic_params['sum_wt_sq'] == 4
        assert basic_params['wt_max'] == 0.25
        assert not np.allclose(basic_params['weights'], orig_weights)

    def test_determine_weights_unequal(self, basic_params):
        basic_params['weight_mode'] = 'useweights'
        basic_params['template'] = np.ones_like(basic_params['data'][0])
        basic_params['weight_frame'] = np.ones_like(basic_params['data'])
        coadd._determine_weighting_method(basic_params)

        for key in ['sum_wt', 'sum_wt_sq', 'wt_max']:
            assert key not in basic_params

        orig_weights = basic_params['weights'].copy()
        coadd._calculate_weights(basic_params)

        assert basic_params['sum_wt'] == 1
        assert basic_params['sum_wt_sq'] == 0.28
        assert basic_params['wt_max'] == 0.4
        assert np.allclose(basic_params['weights'], orig_weights)

    def test_determine_weights_errors(self, basic_params):
        del basic_params['nz']
        param = coadd._verify_inputs(**basic_params)

        # assign even weights to good frames
        param['weights'] = None
        coadd._determine_weighting_method(param)
        assert np.allclose(param['weights'], [0, 1, 1, 1, 1])
        assert param['do_weights'] is True
        assert param['zwt_sum'] == 4

        # zero wtsum raises error
        param['weights'][:] = 0
        with pytest.raises(ValueError) as err:
            coadd._determine_weighting_method(param)
        assert 'All weights are zero' in str(err)

        # directly provided weights do not sum to 1
        param['weights'] = [0, 1, 1, 1, 1]
        param['weight_mode'] = 'use_weights'
        with pytest.raises(ValueError) as err:
            coadd._determine_weighting_method(param)
        assert 'Weights do not add up to 1' in str(err)

    def test_generate_template(self, basic_params, mocker, caplog):
        caplog.set_level(logging.INFO)
        return_value = np.ones_like(basic_params['data'][0])
        mocker.patch.object(make_template, 'make_template',
                            return_value=return_value)
        basic_params['unweighted'] = False
        for key in ['template', 'weight_frame']:
            assert key not in basic_params

        basic_params['std_wt'] = True
        coadd._generate_template(basic_params)

        npt.assert_array_equal(basic_params['template'], return_value)
        npt.assert_array_equal(basic_params['weight_frame'] ** 2,
                               basic_params['variance'])

        basic_params['std_wt'] = False
        coadd._generate_template(basic_params)

        npt.assert_array_equal(basic_params['template'], return_value)
        npt.assert_array_equal(basic_params['weight_frame'],
                               basic_params['flat'])

        # if unweighted, template is directly set to None
        basic_params['unweighted'] = True
        coadd._generate_template(basic_params)
        assert basic_params['template'] is None
        assert 'Using unweighted coadd' not in caplog.text

        # if None is returned for template, unweighted is set
        mocker.patch.object(make_template, 'make_template',
                            return_value=None)
        basic_params['unweighted'] = False
        coadd._generate_template(basic_params)

        assert basic_params['template'] is None
        npt.assert_array_equal(basic_params['weight_frame'],
                               basic_params['flat'])
        assert basic_params['unweighted'] is True
        assert 'Using unweighted coadd' in caplog.text

    def test_calculate_weights(self, basic_params, capsys):
        basic_params.pop('nz')
        orig_param = coadd._verify_inputs(**basic_params)
        coadd._determine_weighting_method(orig_param)
        new_keys = ['sum_wt', 'sum_wt_sq', 'wt_max']
        for key in new_keys:
            assert key not in orig_param

        # unweighted
        param = orig_param.copy()
        param['do_weights'] = True
        param['unweighted'] = True
        coadd._calculate_weights(param)
        assert param['sum_wt'] == 4
        assert param['sum_wt_sq'] == 4
        assert param['wt_max'] == 0.25
        assert np.allclose(param['weights'], [0.25, 0.25, 0, 0.25, 0.25])
        assert capsys.readouterr().err == ''

        # weighted with stdwt
        param = orig_param.copy()
        param['do_weights'] = True
        param['unweighted'] = False
        param['template'] = np.ones_like(param['data'][0])
        param['weight_frame'] = np.ones_like(param['data'])
        param['weights'][0] *= -1
        coadd._calculate_weights(param)
        assert param['sum_wt'] == 3
        assert param['sum_wt_sq'] == 3
        assert param['wt_max'] == 1 / 3
        assert np.allclose(param['weights'], [0, 1 / 3, 0, 1 / 3, 1 / 3])
        assert 'Correlation negative' in capsys.readouterr().err

        # weighted with flat
        param = orig_param.copy()
        param['do_weights'] = True
        param['unweighted'] = False
        param['template'] = np.ones_like(param['data'][0])
        param['std_wt'] = False
        param['weight_frame'] = np.ones_like(param['data'][0])
        param['weights'][:] = 1
        param['data'][0] *= 0
        coadd._calculate_weights(param)
        assert param['sum_wt'] == 4
        assert param['sum_wt_sq'] == 4
        assert param['wt_max'] == 0.25
        assert np.allclose(param['weights'], [0, 0.25, 0.25, 0.25, 0.25])
        assert 'Correlation zero' in capsys.readouterr().err

        # all zero weights
        param['data'] *= 0
        with pytest.raises(ValueError) as err:
            coadd._calculate_weights(param)
        assert 'All weights zero' in str(err)

        # no good data before starting
        param['illum'][:] = 0
        with pytest.raises(ValueError) as err:
            coadd._calculate_weights(param)
        assert 'No good data' in str(err)

    def test_combine_data_neg(self, basic_params, caplog):
        caplog.set_level(logging.INFO)

        basic_params['threshold'] = -2
        coadd._combine_data(basic_params)
        assert 'Performing robust mean' not in caplog.text
        assert 'Weights: ' in caplog.text

        assert basic_params['coadded'].ndim == 2
        assert basic_params['coadded_var'].ndim == 2

    def test_combine_data_pos(self, basic_params, caplog, mocker):
        caplog.set_level(logging.INFO)

        basic_params['threshold'] = 2
        coadd._combine_data(basic_params)
        assert 'Performing robust mean' in caplog.text
        assert 'Weights: ' in caplog.text

        assert basic_params['coadded'].ndim == 2
        assert basic_params['coadded_var'].ndim == 2

    @pytest.mark.parametrize('mode,ctime', [('NOD_OFF_SLIT', 40),
                                            ('NOD_ON_SLIT', 80)])
    def test_update_integration_time(self, basic_params, caplog, mode, ctime):
        caplog.set_level(logging.INFO)
        basic_params['header']['INSTMODE'] = mode

        # todo: update
        keys = ['EXPTIME', 'NEXP']
        for key in keys:
            assert key not in basic_params['header']

        basic_params['sum_wt_sq'] = 9
        basic_params['wt_max'] = 6
        basic_params['header']['NINT'] = 2
        basic_params['header']['BEAMTIME'] = 4

        coadd._update_integration_time(basic_params)

        assert 'Total on-source integration time:' in caplog.text
        assert basic_params['header']['NEXP'] == 10
        assert np.allclose(basic_params['header']['EXPTIME'], ctime)

    def test_coadd_single_order(self, nsb_single_order_hdul, capsys):
        data = nsb_single_order_hdul[0].data
        header = nsb_single_order_hdul[0].header
        variance = nsb_single_order_hdul[1].data ** 2
        flat = nsb_single_order_hdul['FLAT'].data
        illum = nsb_single_order_hdul['FLAT_ILLUMINATION'].data

        # coadd 2 nearly identical frames, nod on slit
        cdata, cvar = coadd.coadd(data, header, flat, variance, illum)
        assert 'Weights: [0.5 0.5]' in capsys.readouterr().out
        assert np.allclose(cdata, np.mean(data, axis=0))
        assert np.allclose(cvar, np.sum(variance, axis=0) / 4)
        assert header['EXPTIME'] == 16

        # coadd 1 frame: returns data
        cdata, cvar = coadd.coadd(np.expand_dims(data[0], axis=0),
                                  header, flat,
                                  np.expand_dims(variance[0], axis=0), illum)
        assert 'Only 1 frame' in capsys.readouterr().out
        assert np.allclose(cdata, data[0])
        assert np.allclose(cvar, variance[0])

    def test_coadd_multi_order(self, nsb_cross_dispersed_hdul, capsys):
        data = nsb_cross_dispersed_hdul[0].data
        header = nsb_cross_dispersed_hdul[0].header
        variance = nsb_cross_dispersed_hdul[1].data ** 2
        flat = nsb_cross_dispersed_hdul['FLAT'].data
        illum = nsb_cross_dispersed_hdul['FLAT_ILLUMINATION'].data

        # coadd 2 nearly identical frames, nod off slit
        cdata, cvar = coadd.coadd(data, header, flat, variance, illum)
        assert 'Weights: [0.5 0.5]' in capsys.readouterr().out
        assert np.allclose(cdata, np.mean(data, axis=0))
        assert np.allclose(cvar, np.sum(variance, axis=0) / 4)
        assert header['EXPTIME'] == 8

        # coadd 1 frame: returns data
        cdata, cvar = coadd.coadd(np.expand_dims(data[0], axis=0),
                                  header, flat,
                                  np.expand_dims(variance[0], axis=0), illum)
        assert 'Only 1 frame' in capsys.readouterr().out
        assert np.allclose(cdata, data[0])
        assert np.allclose(cvar, variance[0])
