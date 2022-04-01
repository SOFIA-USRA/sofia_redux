# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.source_models.sky_dip import SkyDip
from sofia_redux.scan.source_models.sky_dip_model import SkyDipModel


arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
kelvin = units.Unit('Kelvin')
second = units.Unit('second')


@pytest.fixture
def skydip_source(skydip_scan):
    reduction = Reduction('example')
    source = SkyDip(reduction.info, reduction)
    scans = [skydip_scan]
    source.configuration.parse_key_value('skydip.grid', '900.0')
    source.configuration.parse_key_value('skydip.signal', 'obs-channels')
    source.configuration.parse_key_value('skydip.mode', '0')
    source.configuration.parse_key_value('skydip.fit',
                                         'tau,offset,kelvin,tsky')
    source.configuration.parse_key_value('skydip.elrange', '0:90')
    source.configuration.parse_key_value('skydip.uniform', 'True')
    source.create_from(scans)
    scan = source.scans[0]
    scan.validate()
    integration = scan[0]
    source.add_integration(integration)
    source.end_accumulation()
    return source


@pytest.fixture
def configured_model(skydip_source):
    source = skydip_source.copy()
    source.configuration.parse_key_value('skydip.tsky', '275.0')
    source.tamb = 276 * kelvin
    source.tamb_weight = 1 * second
    model = SkyDipModel()
    model.set_configuration(source.configuration)
    return model


@pytest.fixture
def fitted_model(configured_model, skydip_source):
    model = configured_model.copy()
    model.fit(skydip_source)
    return model


def test_class():
    guess = SkyDipModel.default_initial_guess
    bounds = SkyDipModel.default_bounds
    assert guess['tsky'] == 273
    assert np.isnan(guess['offset'])
    assert np.isnan(guess['kelvin'])
    assert guess['tau'] == 1
    assert bounds['tsky'] == [0.0, np.inf]
    assert bounds['offset'] == [-np.inf, np.inf]
    assert bounds['kelvin'] == [0.0, np.inf]
    assert bounds['tau'] == [0.0, 10.0]


def test_init():
    model = SkyDipModel()
    assert model.configuration is None
    assert model.initial_guess == SkyDipModel.default_initial_guess
    assert model.bounds == SkyDipModel.default_bounds
    assert model.fit_for is None
    assert not model.has_converged
    assert model.data_unit == units.Unit('count')
    assert model.use_points == 0
    assert not model.uniform_weights
    assert model.el_range.min == -np.inf and model.el_range.max == np.inf
    assert model.parameters is None
    assert model.errors is None
    assert np.isnan(model.rms)
    assert model.fitted_values is None
    assert model.elevation is None
    assert model.data is None
    assert model.sigma is None
    assert model.p_opt is None
    assert model.p_cov is None


def test_copy(configured_model):
    model = configured_model
    model.data = np.arange(10)
    new_model = model.copy()
    assert np.allclose(model.data, new_model.data)
    assert model.data is not new_model.data


def test_set_configuration(skydip_source):
    configuration = skydip_source.configuration.copy()
    model = SkyDipModel()
    with pytest.raises(ValueError) as err:
        model.set_configuration(None)
    assert "Configuration must be" in str(err.value)

    configuration.parse_key_value('skydip.elrange', '0:90')
    configuration.parse_key_value('skydip.uniform', 'True')
    configuration.parse_key_value('skydip.fit', 'tau,offset,data2k,tsky')
    model.set_configuration(configuration)
    assert model.el_range.min == 0 * degree
    assert model.el_range.max == 90 * degree
    assert model.uniform_weights
    assert model.fit_for == ['kelvin', 'offset', 'tau', 'tsky']

    configuration.parse_key_value('skydip.uniform', 'False')
    del configuration['skydip.fit']
    model.set_configuration(configuration)
    assert not model.uniform_weights
    assert model.fit_for == ['kelvin', 'offset', 'tau']


def test_init_parameters(configured_model, skydip_source):
    model = configured_model.copy()
    source = skydip_source
    source.tamb = 273 * kelvin
    source.tamb_weight = 1 * second
    model.fit_for.remove('kelvin')
    model.init_parameters(source)
    assert 'kelvin' in model.fit_for
    assert model.initial_guess['tsky'] == 275
    assert np.isclose(model.initial_guess['offset'], 27.9, atol=0.1)
    assert np.isclose(model.initial_guess['kelvin'], 0.34, atol=0.01)
    assert np.isclose(model.initial_guess['tau'], 1.0)

    model = configured_model.copy()
    del model.configuration['skydip.tsky']
    model.initial_guess['kelvin'] = 2 * kelvin
    model.init_parameters(source)
    assert model.initial_guess['tsky'] == 273
    assert np.isclose(model.initial_guess['offset'], 27.9, atol=0.1)
    assert np.isclose(model.initial_guess['kelvin'], 2)
    assert np.isclose(model.initial_guess['tau'], 0.0371, atol=1e-3)

    model = configured_model.copy()
    model.initial_guess['kelvin'] = -1
    model.init_parameters(source)
    assert model.initial_guess['tau'] == 0.1
    model = configured_model.copy()
    model.initial_guess['kelvin'] = 1e-3
    model.init_parameters(source)
    assert model.initial_guess['tau'] == 1

    source.data.fill(np.nan)
    model = configured_model.copy()
    model.init_parameters(source)
    assert model.initial_guess['kelvin'] == 1
    assert model.initial_guess['offset'] == 0


def test_fit(configured_model, skydip_source):
    model = configured_model.copy()
    model.fit_for = ['kelvin', 'offset', 'tau']  # not tsky
    source = skydip_source.copy()
    assert model.uniform_weights
    assert model.el_range is not None
    model.fit(source)
    assert np.isclose(np.rad2deg(model.elevation[0]), 10.125)
    assert model.use_points == 281
    assert model.has_converged
    assert model.sigma is None
    assert np.allclose(model.data, source.data[source.data != 0])
    assert np.allclose(model.data, model.fitted_values, atol=1)
    names = ['tau', 'offset', 'kelvin', 'tsky']
    for parameter in names:
        assert parameter in model.parameters
        assert parameter in model.errors

    assert isinstance(model.p_opt, np.ndarray) and model.p_opt.shape == (4,)
    assert isinstance(model.p_cov, np.ndarray) and model.p_cov.shape == (4, 4)

    assert np.isclose(model.rms, 0.05, atol=1e-1)

    model.el_range = None
    model.uniform_weights = False
    model.fit(source)
    assert isinstance(model.sigma, np.ndarray)
    assert model.has_converged
    assert np.allclose(model.data, model.fitted_values, atol=1)


def test_fit_elevation(fitted_model):
    model = fitted_model.copy()
    result = model.fit_elevation(45 * degree)
    assert np.isclose(result, -9, atol=1)
    model.p_opt = None
    assert np.isnan(model.fit_elevation(45 * degree))


def test_value_at():
    result = SkyDipModel.value_at(np.pi / 3, 0.1, 0, 1, 273)
    assert np.isclose(result, 29.771400, atol=1e-6)


def test_get_parameter_string(fitted_model):
    model = fitted_model.copy()
    s = model.get_parameter_string('kelvin')
    assert 'kelvin =' in s and '+/-' in s and 'ct' in s
    model.errors['kelvin'] = np.nan
    s = model.get_parameter_string('kelvin')
    assert '+/-' not in s
    assert model.get_parameter_string('foo') is None
    model.has_converged = False
    assert model.get_parameter_string('kelvin') is None


def test_get_parameter_format():
    assert SkyDipModel.get_parameter_format('tau') == '%.3f'
    assert SkyDipModel.get_parameter_format('tsky') == '%.1f'
    assert SkyDipModel.get_parameter_format('kelvin') == '%.3e'
    assert SkyDipModel.get_parameter_format('foo') == '%.3e'


def test_get_parameter_unit(configured_model):
    model = configured_model
    assert model.get_parameter_unit('tsky') == kelvin
    assert model.get_parameter_unit('kelvin') == units.Unit('count')
    assert model.get_parameter_unit('foo') is None


def test_str(fitted_model):
    model = fitted_model
    s = str(model)
    assert 'tau =' in s and 'kelvin =' in s
    assert 'offset =' in s and 'tsky =' in s
    assert 'K rms]' in s
    model.has_converged = False
    s = str(model)
    assert s == ''
