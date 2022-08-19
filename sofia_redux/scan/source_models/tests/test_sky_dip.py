# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import os
import pytest

from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.source_models.sky_dip import SkyDip
from sofia_redux.scan.source_models.sky_dip_model import SkyDipModel


arcsec = units.Unit('arcsec')
degree = units.Unit('degree')
kelvin = units.Unit('Kelvin')
second = units.Unit('second')


@pytest.fixture
def example_reduction():
    return Reduction('example')


@pytest.fixture
def basic_source(example_reduction):
    source = SkyDip(example_reduction.info, example_reduction)
    return source


@pytest.fixture
def initialized_source(basic_source, skydip_scan):
    source = basic_source
    scans = [skydip_scan]
    source.configuration.parse_key_value('skydip.grid', '900.0')
    source.configuration.parse_key_value('skydip.signal', 'obs-channels')
    source.configuration.parse_key_value('skydip.mode', '0')
    source.configuration.parse_key_value('skydip.fit',
                                         'tau,offset,kelvin,tsky')
    source.configuration.parse_key_value('skydip.elrange', '0:90')
    source.configuration.parse_key_value('skydip.uniform', 'True')
    source.create_from(scans)
    return source


@pytest.fixture
def data_source(initialized_source):
    source = initialized_source
    scan = source.scans[0]
    scan.validate()
    integration = scan[0]
    source.add_integration(integration)
    source.end_accumulation()
    return source


def test_init(example_reduction):
    reduction = example_reduction
    info = reduction.info
    source = SkyDip(info, reduction=reduction)
    assert source.data is None
    assert source.weight is None
    assert source.resolution == 0.0
    assert source.tamb == 0 * kelvin
    assert source.tamb_weight == 0 * second
    assert source.signal_name == 'obs-channels'
    assert source.signal_index == 0
    assert isinstance(source.model, SkyDipModel)
    assert source.reduction is reduction
    assert source.info is reduction.info


def test_copy(basic_source):
    source = basic_source
    source.data = np.arange(10)
    source2 = source.copy(with_contents=True)
    assert np.allclose(source2.data, source.data)
    source3 = source.copy(with_contents=False)
    assert source3.data is None
    assert source3.reduction is source.reduction


def test_clear_all_memory(data_source):
    source = data_source.copy()
    source.clear_all_memory()
    assert source.data is None
    assert source.weight is None


def test_logging_id(basic_source):
    assert basic_source.logging_id == 'skydip'


def test_get_unit(basic_source):
    assert basic_source.get_unit() == kelvin


def test_get_table_entry(basic_source):
    source = basic_source
    parameters = ['tau', 'dtau', 'kelvin', 'dkelvin', 'tsky', 'dtsky']
    for parameter in parameters:
        assert source.get_table_entry(parameter) is None
    source.model.parameters = {'tau': 1, 'kelvin': 2, 'tsky': 3}
    source.model.errors = {'tau': 4, 'kelvin': 5, 'tsky': 6}
    assert source.get_table_entry('tau') == 1
    assert source.get_table_entry('kelvin') == 2
    assert source.get_table_entry('tsky') == 3
    assert source.get_table_entry('dtau') == 4
    assert source.get_table_entry('dkelvin') == 5
    assert source.get_table_entry('dtsky') == 6
    assert source.get_table_entry('foo') is None


def test_clear_content(basic_source):
    source = basic_source
    source.weight = np.arange(10)
    source.data = np.arange(10)
    source.tamb = 1 * kelvin
    source.tamb_weight = 1 * second
    source.clear_content()
    assert np.allclose(source.weight, 0)
    assert np.allclose(source.data, 0)
    assert source.tamb == 0 * kelvin
    assert source.tamb_weight == 0 * second
    source.data = None
    source.weight = None
    source.clear_content()
    assert source.data is None
    assert source.weight is None


def test_get_average_temp_with(basic_source):
    source = basic_source
    source.tamb = 2 * kelvin
    source.tamb_weight = 1 * second
    source.average_temp_with(4 * kelvin, 3 * second)
    assert source.tamb == 3.5 * kelvin
    assert source.tamb_weight == 4 * second
    source.average_temp_with(5 * kelvin, 0 * second)
    assert source.tamb == 3.5 * kelvin
    assert source.tamb_weight == 4 * second


def test_create_from(basic_source, skydip_scan):
    source = basic_source
    scans = [skydip_scan]
    source.configuration.parse_key_value('skydip.grid', '1800.0')
    source.configuration.parse_key_value('skydip.signal', 'foo')
    source.configuration.parse_key_value('skydip.mode', '1')
    source.create_from(scans)
    assert source.scans == scans
    assert source.resolution == 1800 * arcsec
    assert source.signal_name == 'foo'
    assert source.signal_index == 1
    expected = np.zeros(180, dtype=float)
    assert np.allclose(source.data, expected)
    assert np.allclose(source.weight, expected)
    del source.configuration['skydip.grid']
    source.create_from(scans)
    assert source.resolution == 900 * arcsec
    expected = np.zeros(360, dtype=float)
    assert np.allclose(source.data, expected)
    assert np.allclose(source.weight, expected)


def test_get_bin(initialized_source):
    source = initialized_source
    assert source.get_bin(np.nan * degree) == -1
    assert source.get_bin(np.inf * degree) == -1
    assert source.get_bin(45 * degree) == 180
    assert np.allclose(source.get_bin(np.arange(3) * degree), [0, 4, 8])


def test_get_elevation(initialized_source):
    source = initialized_source
    assert np.allclose(source.get_elevation(np.arange(3)),
                       [450, 1350, 2250] * arcsec)


def test_add_model_data(initialized_source):
    source1 = initialized_source.copy()
    source2 = initialized_source.copy()
    source1.tamb = 270 * kelvin
    source1.tamb_weight = 1 * second
    source2.tamb = 300 * kelvin
    source2.tamb_weight = 1 * second
    data = source1.data
    source2.data = None
    source1.add_model_data(source2)
    assert source1.tamb == 285 * kelvin
    assert source1.tamb_weight == 2 * second
    source2.tamb_weight = 0 * second  # No more averaging tamb
    source2.data = np.ones_like(data)
    source2.weight = np.full_like(data, 2.0)
    source1.data = None
    source1.add_model_data(source2)
    assert np.allclose(source1.data, source2.data)
    assert np.allclose(source1.weight, source2.weight)

    source2.data = np.ones(25)
    with pytest.raises(ValueError) as err:
        source1.add_model_data(source2)
    assert 'SkyDip data shapes do not match' in str(err.value)

    source2.data = np.full_like(source1.data, 2.0)
    source2.weight = np.full_like(source1.weight, 2.0)
    source2.weight[0] = 0
    source1.add_model_data(source2)
    assert source1.data[0] == 1
    assert np.allclose(source1.data[1:], 1.5)
    assert source1.weight[0] == 2
    assert np.allclose(source1.weight[1:], 4)


def test_add_integration(initialized_source, capsys):
    source = initialized_source
    integration = source.scans[0][0]
    source.add_integration(integration)
    assert "Cannot decorrelate sky channels" in capsys.readouterr().err
    assert np.allclose(source.data[:40], 0)
    assert np.allclose(source.data[321:], 0)
    assert np.allclose(source.weight[:40], 0)
    assert np.allclose(source.weight[321:], 0)
    assert np.allclose(source.data[40:45],
                       [14461, 28437, 27812, 13680, 26925], atol=1)
    assert np.allclose(source.weight[40:45], [121, 242, 242, 121, 242])
    source.data.fill(0)
    source.weight.fill(0)
    integration.signals = {}
    integration.validate()
    source.add_integration(integration)
    assert np.allclose(source.data[:40], 0)
    assert np.allclose(source.data[321:], 0)
    assert np.allclose(source.weight[:40], 0)
    assert np.allclose(source.weight[321:], 0)
    assert np.allclose(source.data[40:45],
                       [4.67859015e-11, 3.99544009e-11, 5.30851266e-11,
                        -6.67554206e-12, 2.18205037e-11], rtol=1e-2)
    assert np.allclose(source.weight[40:45], [121, 242, 242, 121, 242])

    source.signal_index = 100
    with pytest.raises(ValueError) as err:
        source.add_integration(integration)
    assert 'Cannot retrieve signal index' in str(err.value)

    source.signal_name = 'foo'
    with pytest.raises(ValueError) as err:
        source.add_integration(integration)
    assert 'foo not found in integration' in str(err.value)


def test_set_base(basic_source):
    source = basic_source
    source.set_base()  # Does nothing


def test_get_source_name(basic_source):
    assert basic_source.get_source_name() == 'SkyDip'


def test_end_accumulation(data_source):
    source = data_source.copy()
    source.data *= source.weight
    source.end_accumulation()
    assert np.allclose(source.data[40:45],
                       [74.48, 72.48, 69.89, 68.03, 66.23], atol=0.1)
    assert np.allclose(source.data[:40], 0)
    source.data = None
    source.end_accumulation()
    assert source.data is None


def test_process_scan(data_source):
    source = data_source.copy()
    expected_data = source.data.copy()
    source.data *= source.weight

    class DummyInfo(object):
        @staticmethod
        def get_ambient_kelvins():
            return 400 * kelvin

    class DummyScan(object):
        def __init__(self):
            self.info = DummyInfo()

        @staticmethod
        def get_observing_time():
            return 10 * second

    scan = DummyScan()
    source.tamb = 300 * kelvin
    source.tamb_weight = 10 * second
    source.process_scan(scan)
    assert source.tamb == 350 * kelvin
    assert source.tamb_weight == 20 * second
    assert np.allclose(source.data, expected_data)


def test_count_points(data_source):
    source = data_source.copy()
    assert source.count_points() == 281
    source.weight = None
    assert source.count_points() == 0


def test_sync_integration(data_source):
    integration = data_source.scans[0][0]
    data_source.sync_integration(integration)  # Does nothing


def test_get_signal_range(data_source):
    source = data_source.copy()
    r = source.get_signal_range()
    assert np.isclose(r.min, -18.672, atol=1e-2)
    assert np.isclose(r.max, 74.484, atol=1e-2)
    source.data = None
    r = source.get_signal_range()
    assert r.min == -np.inf and r.max == np.inf


def test_get_elevation_range(data_source):
    source = data_source.copy()
    r = source.get_elevation_range()
    assert np.isclose(r.min, 10.125 * degree)
    assert np.isclose(r.max, 80.125 * degree)


def test_get_air_mass_range(data_source):
    r = data_source.get_air_mass_range()
    assert np.isclose(r.min, 1.015039, atol=1e-5)
    assert np.isclose(r.max, 5.688403, atol=1e-5)


def test_fit(data_source):
    source = data_source.copy()
    source.fit(source.model)
    p = source.model.parameters
    assert np.isclose(p['tau'], 0.098, atol=1e-1)
    assert np.isclose(p['offset'], -45.16, atol=1e-1)
    # The following values may vary widely depending on the Python architecture
    # of the optimization routine.
    assert np.isfinite(p['kelvin'])
    assert np.isfinite(p['tsky'])


def test_write(data_source, tmpdir):
    source = data_source.copy()
    path = str(tmpdir.mkdir('test_write'))
    source.configuration.work_path = path
    source.write()
    files = os.listdir(path)
    assert len(files) == 1 and files[0] == 'SkyDip.Simulation.1.dat'
    filename = os.path.join(path, files[0])
    with open(filename, 'r') as f:
        lines = f.readlines()

    assert lines[0].startswith('# tau')
    assert lines[1].startswith('# offset')
    assert lines[2].startswith('# kelvin')
    assert lines[3].startswith('# tsky')

    for line in lines[4:]:
        if line.startswith('0.125'):
            assert '...' in line
        if line.startswith('10.125'):
            assert '...' not in line  # Valid result
            break
    else:  # pragma: no cover
        raise ValueError('Test failed')


def test_create_plot(basic_source):
    basic_source.create_plot('foo')  # Does nothing


def test_no_parallel(basic_source):
    basic_source.no_parallel()  # Does nothing


def test_set_parallel(basic_source):
    basic_source.set_parallel(1)  # Does nothing


def test_process(basic_source):
    basic_source.process()  # Does nothing


def test_is_valid(data_source):
    source = data_source.copy()
    assert source.is_valid()
    source.weight.fill(0)
    assert not source.is_valid()


def test_set_executor(basic_source):
    basic_source.set_executor(None)  # Does nothing


def test_get_parallel(basic_source):
    assert basic_source.get_parallel() == 1


def test_get_reference(basic_source):
    assert basic_source.get_reference() is None
