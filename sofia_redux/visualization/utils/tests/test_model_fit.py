#  Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
from astropy.modeling import models, core, functional_models
import matplotlib.figure as mpf

from sofia_redux.visualization.utils.eye_error import EyeError
from sofia_redux.visualization.utils import model_fit


class TestModelFit(object):

    def test_init_blank(self):
        mf = model_fit.ModelFit()
        assert mf.id_tag.startswith('model_fit')
        assert mf.model_id == ''
        assert mf.feature == ''
        assert mf.background == ''
        assert mf.fit_type == ['gauss']
        assert mf.order == 0
        assert mf.status == 'pass'
        assert mf.fit is None
        assert mf.axis_names == ['x', 'y']
        assert mf.units == {'x': '', 'y': ''}
        assert mf.limits == {'lower': None, 'upper': None}
        assert mf.fields == {'x': '', 'y': ''}
        assert mf.dataset == {'x': None, 'y': None}
        assert mf.axis is None
        assert mf.visible

        mf2 = model_fit.ModelFit()
        assert mf2.id_tag.startswith('model_fit')
        tag1 = mf.id_tag.split('_')[-1]
        tag2 = mf2.id_tag.split('_')[-1]
        assert int(tag2) - int(tag1) == 1

    def test_init_param(self, mocker):
        load_mock = mocker.patch.object(model_fit.ModelFit, 'load_parameters')
        model_fit.ModelFit(dict())
        assert load_mock.called_once()

    def test_get_id(self):
        mf = model_fit.ModelFit()
        id_tag = mf.get_id()
        assert id_tag == mf.id_tag

    def test_load_parameters(self, gauss_params):
        mf = model_fit.ModelFit()
        mf.load_parameters(gauss_params)

        assert mf.model_id == 'model_id_name'
        assert mf.order == 1
        assert isinstance(mf.fit, core.CompoundModel)
        assert mf.fit_type == ['gauss', 'linear']

    def test_load_parameters_single_model(self, single_gauss_params):
        mf = model_fit.ModelFit()
        mf.load_parameters(single_gauss_params)

        assert mf.model_id == 'model_id_name'
        assert mf.order == 1
        assert isinstance(mf.fit, functional_models.Gaussian1D)
        assert mf.fit_type == ['gauss']

    def test_get_set_fit_type(self):
        mf = model_fit.ModelFit()
        feature = 'lorentz'
        background = 'constant'

        mf.set_fit_type(fit_type=feature)
        assert mf.fit_type == [feature]

        mf.set_fit_type(fit_type=[feature, background])
        assert mf.fit_type == [feature, background]
        assert mf.get_fit_types() == [feature, background]
        assert mf.feature == ''
        assert mf.background == ''

        mf.set_fit_type(feature=feature, background=background)
        assert mf.fit_type == [feature, background]
        assert mf.feature == feature
        assert mf.background == background

        with pytest.raises(EyeError):
            mf.set_fit_type()

    def test_get_set_feature(self):
        mf = model_fit.ModelFit()

        assert mf.feature == ''
        feature = 'lorentz'
        mf.set_feature(feature)
        assert mf.feature == feature
        assert mf.get_feature() == feature

    def test_get_set_background(self):
        mf = model_fit.ModelFit()

        assert mf.background == ''
        background = 'lorentz'
        mf.set_background(background)
        assert mf.background == background
        assert mf.get_background() == background

    def test_get_set_fields(self):
        mf = model_fit.ModelFit()

        assert mf.fields == {'x': '', 'y': ''}
        fields = {'x': 'wave', 'y': 'flux', 'z': 'not'}
        mf.set_fields(fields)
        assert mf.fields == fields
        assert mf.get_fields() == fields
        assert mf.get_fields('x') == fields['x']
        assert mf.get_fields('w') is None

    def test_get_set_axis(self):
        mf = model_fit.ModelFit()

        assert mf.axis is None
        axis = mpf.Figure().subplots(1, 1)
        mf.set_axis(axis)
        assert mf.axis == axis
        assert mf.get_axis() == axis

    def test_get_set_status(self):
        mf = model_fit.ModelFit()

        assert mf.status == 'pass'
        status = 'ongoing'
        mf.set_status(status)
        assert mf.status == status
        assert mf.get_status() == status

    def test_get_set_model_id(self):
        mf = model_fit.ModelFit()

        assert mf.model_id == ''
        model_id = 'name'
        mf.set_model_id(model_id)
        assert mf.model_id == model_id
        assert mf.get_model_id() == model_id

    def test_get_set_order(self):
        mf = model_fit.ModelFit()

        assert mf.order == 0
        order = 2
        mf.set_order(order)
        assert mf.order == order
        assert mf.get_order() == order

    def test_get_set_dataset(self):
        mf = model_fit.ModelFit()

        assert mf.dataset == {'x': None, 'y': None}
        dataset = {'x': np.arange(5), 'y': np.arange(5)}
        mf.set_dataset(dataset)
        assert mf.dataset == dataset
        assert mf.get_dataset() == dataset

        dataset = {'x': np.arange(10), 'y': np.arange(10)}
        mf.set_dataset(x=dataset['x'], y=dataset['y'])
        assert mf.get_dataset() == dataset

        with pytest.raises(EyeError):
            mf.set_dataset()

        with pytest.raises(EyeError):
            mf.set_dataset(x=dataset['x'])

    def test_get_set_visibility(self):
        mf = model_fit.ModelFit()

        assert mf.visible
        mf.set_visibility(False)
        assert not mf.visible
        assert not mf.get_visibility()

    def test_get_set_fit(self):
        mf = model_fit.ModelFit()

        assert mf.fit is None
        assert mf.get_fit() is None
        fit = models.Gaussian1D()
        mf.set_fit(fit)
        assert mf.fit != fit
        assert mf.get_fit() != fit
        assert str(mf.get_fit()) == str(fit)

    def test_get_set_units(self):
        mf = model_fit.ModelFit()

        assert mf.units == {'x': '', 'y': ''}
        units = {'x': 'nm', 'y': 'W/m**2'}
        mf.set_units(units)
        assert mf.units == units
        assert mf.get_units() == units
        assert mf.get_units('x') == units['x']
        assert mf.get_units('z') is None

    def test_set_limits(self):
        mf = model_fit.ModelFit()

        assert mf.limits == {'lower': None, 'upper': None}
        limits = {'lower': 10, 'upper': 20}
        mf.set_limits(limits)
        assert mf.limits == limits
        assert mf.get_limits() == limits

        lower = 15
        mf.set_limits(lower, 'lower')
        assert mf.get_limits('lower') == lower

        limits = [[1, 2], [3, 4]]
        mf.set_limits(limits)
        assert mf.get_limits() == {'lower': 1, 'upper': 3}

        assert mf.get_limits('other') is None

    @pytest.mark.parametrize('fit,result',
                             [(models.Gaussian1D(), 'gauss'),
                              (models.Moffat1D(), 'moffat'),
                              (models.Linear1D(), 'linear'),
                              (models.Const1D(), 'constant'),
                              (models.Lorentz1D(), 'UNKNOWN')])
    def test_determine_fit_type(self, fit, result):
        out = model_fit.ModelFit._determine_fit_type(fit)
        assert out == result

    def test_parameters_as_string(self, gauss_params):
        mf = model_fit.ModelFit(gauss_params)
        ax = gauss_params['model_id_name'][1]['axis']

        params = {'model_id': 'model_id_name', 'order': '1',
                  'x_field': 'wavelength [nm]', 'y_field': 'flux [Jy]',
                  'lower_limit': '5', 'upper_limit': '15',
                  'type': 'gauss, linear', 'axis': ax,
                  'visible': False, 'baseline': '2',
                  'mean': '0', 'stddev': '0.2', 'amplitude': '1',
                  'slope': '0.5', 'intercept': '2',
                  'fwhm': '0.47096', 'mid_point': '0'
                  }

        output = mf.parameters_as_string()
        assert output == params

    def test_parameters_as_dict(self, gauss_params):
        mf = model_fit.ModelFit(gauss_params)
        ax = gauss_params['model_id_name'][1]['axis']
        fit = gauss_params['model_id_name'][1]['fit']

        params = {'model_id': 'model_id_name', 'order': 1,
                  'x_field': 'wavelength', 'y_field': 'flux',
                  'x_unit': 'nm', 'y_unit': 'Jy',
                  'lower_limit': 5, 'upper_limit': 15,
                  'type': ['gauss', 'linear'], 'axis': ax,
                  'visible': False, 'fit': fit,
                  'baseline': 2.0, 'fwhm': fit[0].fwhm, 'mid_point': 0.0,
                  'slope': 0.5, 'intercept': 2.0, 'mean': 0.0,
                  'stddev': 0.2, 'amplitude': 1.0
                  }

        output = mf.parameters_as_dict()
        assert output == params

    def test_parameters_as_list(self, gauss_params):
        mf = model_fit.ModelFit(gauss_params)

        params = ['model_id_name', 1, 'wavelength', 'flux',
                  'nm', 'Jy', '0', '0.2', '1',
                  '2', '0.5', '2', '0',
                  '0.47096', 5, 15, False]

        output = mf.parameters_as_list()
        assert output == params

    def test_parameters_as_html(self, gauss_params):
        mf = model_fit.ModelFit(gauss_params)

        params = ('<html><br>Last fit status: '
                  '<span style="color: green">Pass</span><br>'
                  'Parameters: <pre><br>'
                  '  model_id: model_id_name<br>'
                  '  order: 1<br>'
                  '  x_field: wavelength [nm]<br>'
                  '  y_field: flux [Jy]<br>'
                  '  lower_limit: 5<br>'
                  '  upper_limit: 15<br>'
                  '  type: gauss, linear<br>'
                  '  baseline: 2<br>'
                  '  mid_point: 0<br>'
                  '  fwhm: 0.47096<br>'
                  '  mean: 0<br>'
                  '  stddev: 0.2<br>'
                  '  amplitude: 1<br>'
                  '  intercept: 2<br>'
                  '  slope: 0.5<br></pre></html>')

        output = mf.parameters_as_html()
        assert output == params

        # failed status
        mf.status = 'Bad fit'
        output = mf.parameters_as_html()
        assert '<span style="color: red">Bad Fit</span>' in output

    def test_get_mid_point(self, gauss_fit, single_gauss_fit,
                           moffat_fit, lorentz_fit):
        mf = model_fit.ModelFit()
        value = mf.get_mid_point()
        assert np.isnan(value)

        mf.set_fit(gauss_fit)
        value = mf.get_mid_point()
        assert value == 0

        mf.set_fit(moffat_fit)
        value = mf.get_mid_point()
        assert value == 5

        mf.set_fit(single_gauss_fit)
        value = mf.get_mid_point()
        assert value == 0

        mf.set_fit(lorentz_fit)
        value = mf.get_mid_point()
        assert np.isnan(value)

    def test_get_fwhm(self, gauss_fit, moffat_fit, single_gauss_fit,
                      lorentz_fit):
        mf = model_fit.ModelFit()
        value = mf.get_fwhm()
        assert np.isnan(value)

        mf.set_fit(gauss_fit)
        value = mf.get_fwhm()
        assert np.allclose(value, 0.471, atol=1e-3)

        mf.set_fit(moffat_fit)
        value = mf.get_fwhm()
        assert np.allclose(value, 33.407, atol=1e-3)

        mf.set_fit(single_gauss_fit)
        value = mf.get_fwhm()
        assert np.allclose(value, 0.471, atol=1e-3)

        mf.set_fit(lorentz_fit)
        value = mf.get_fwhm()
        assert np.isnan(value)

    def test_get_baseline(self, gauss_fit, moffat_fit, single_gauss_fit,
                          gauss_const_fit, mocker):
        mf = model_fit.ModelFit()
        value = mf.get_baseline()
        assert np.isnan(value)

        mf.set_fit(gauss_fit)
        value = mf.get_baseline()
        assert value == 2

        mf.set_fit(moffat_fit)
        value = mf.get_baseline()
        assert value == 4.5

        mf.set_fit(single_gauss_fit)
        value = mf.get_baseline()
        assert value == 0

        mf.set_fit(gauss_const_fit)
        value = mf.get_baseline()
        assert value == 2

        # bad midpoint returns nan
        mocker.patch.object(mf, 'get_mid_point',
                            return_value=np.nan)
        value = mf.get_baseline()
        assert np.isnan(value)

    def test_scale_parameters(self, gauss_fit, moffat_fit,
                              gauss_const_fit, single_gauss_fit):
        xs, ys = 2.0, 3.0
        x_keys = ['lower_limit', 'upper_limit',
                  'fwhm', 'mid_point']
        y_keys = ['baseline', 'intercept']
        xy_keys = ['slope']

        mf = model_fit.ModelFit()

        # no-op without fit
        mf.scale_parameters(xs, ys)

        models = [gauss_fit, moffat_fit, gauss_const_fit, single_gauss_fit]
        mtypes = [['gauss', 'linear'], ['moffat', 'linear'],
                  ['gauss', 'constant'], ['gauss']]
        for model, mtype in zip(models, mtypes):
            mf.set_fit(model)
            mf.set_fit_type(mtype)

            par1 = mf.parameters_as_dict()

            mf.scale_parameters(xs, ys)
            par2 = mf.parameters_as_dict()
            for key in x_keys:
                if key in par1 and par1[key] is not None:
                    assert par2[key] == xs * par1[key]
            for key in y_keys:
                if key in par1 and par1[key] is not None:
                    assert par2[key] == ys * par1[key]
            for key in xy_keys:
                if key in par1 and par1[key] is not None:
                    assert par2[key] == ys * par1[key] / xs
