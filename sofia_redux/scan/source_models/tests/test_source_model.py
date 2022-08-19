# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from copy import deepcopy
import numpy as np
import pytest
import queue

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.source_models.source_model import SourceModel
from sofia_redux.scan.reduction.reduction import Reduction
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo


class FunctionalSourceModel(SourceModel):  # pragma: no cover

    def __init__(self, info, reduction=None):
        super().__init__(info, reduction=reduction)
        self.synced_integrations = 0

    def count_points(self):
        return 1

    def get_source_name(self):
        return 'test'

    def get_unit(self):
        return 1 * units.Unit('Jy')

    def clear_content(self):
        pass

    def is_valid(self, valid=True):
        return valid

    def add_model_data(self, source_model, weight=1.0):
        pass

    def add_integration(self, integration):
        pass

    def process(self):
        pass

    def process_scan(self, scan):
        pass

    def sync_integration(self, integration, signal_mode=None):
        self.synced_integrations += 1

    def set_base(self):
        pass

    def write(self, path):
        pass

    def get_reference(self):
        return None


@pytest.fixture
def example_reduction():
    return Reduction('example')


@pytest.fixture
def basic_source(example_reduction):
    source = FunctionalSourceModel(example_reduction.info,
                                   reduction=example_reduction)
    return source


@pytest.fixture
def populated_source(basic_source, populated_scan):
    source = basic_source.copy()
    source.scans = [populated_scan]
    return source


def test_init(example_reduction):
    reduction = example_reduction
    info = reduction.info
    source = FunctionalSourceModel(info)
    assert source.info is info
    assert source.scans is None
    assert source.hdul is None
    assert source.id == ''
    assert source.generation == 0
    assert source.integration_time == 0 * units.Unit('second')
    assert source.enable_level and source.enable_bias
    assert source.process_brief is None
    assert source.reduction is None
    source = FunctionalSourceModel(info, reduction=reduction)
    assert source.reduction is reduction


def test_copy(basic_source):
    source = basic_source
    source.scans = [1, 2, 3]
    source.process_brief = 1
    source.integration_time = 1 * units.Unit('second')
    source2 = source.copy()
    assert source2.reduction is source.reduction
    assert source2.scans is source.scans
    assert source2.integration_time == source.integration_time
    assert source2.integration_time is not source.integration_time
    assert source2.process_brief is None

    source2 = source.copy(with_contents=False)
    assert source2.scans is source.scans
    assert source2.reduction is source.reduction
    assert source2.integration_time != source.integration_time


def test_clear_all_memory(basic_source):
    source = basic_source.copy()
    source.info = 1
    source.scans = 2
    source.hdul = 3
    source.generation = 4
    source.clear_all_memory()
    assert source.info is None
    assert source.scans is None
    assert source.hdul is None
    assert source.generation == 0


def test_referenced_attributes(basic_source):
    assert 'scans' in basic_source.referenced_attributes
    assert 'reduction' in basic_source.referenced_attributes


def test_recycler():
    c = FunctionalSourceModel
    assert c.recycler is None
    c.set_recycler_capacity(2)
    assert isinstance(c.recycler, queue.Queue)
    assert c.recycler.maxsize == 2
    assert c.recycler.empty()
    c.clear_recycler()
    assert c.recycler.empty()
    c.recycler.put(1)
    assert not c.recycler.empty()
    c.clear_recycler()
    assert c.recycler.empty()
    c.set_recycler_capacity(0)
    assert c.recycler is None


def test_get_recycled_clean_local_copy(basic_source):
    source = basic_source
    c = FunctionalSourceModel
    assert c.recycler is None
    source2 = source.get_recycled_clean_local_copy()
    assert isinstance(source2, FunctionalSourceModel)
    c.set_recycler_capacity(1)
    assert isinstance(c.recycler, queue.Queue)
    c.recycler.put(source2)
    assert not c.recycler.empty()
    source3 = source.get_recycled_clean_local_copy()
    assert source3 is source2
    assert c.recycler.empty()
    c.set_recycler_capacity(0)
    assert c.recycler is None


def test_get_clean_local_copy(basic_source):
    source = basic_source
    source2 = source.get_clean_local_copy()
    assert source is not source2
    assert source.reduction is source2.reduction


def test_recycle(basic_source, capsys):
    c = FunctionalSourceModel
    source = basic_source
    assert c.recycler is None
    source.recycle()
    assert c.recycler is None
    c.set_recycler_capacity(1)
    source.recycle()
    source.recycle()  # Cannot place because queue too small
    assert "Source recycler overflow" in capsys.readouterr().err

    assert isinstance(c.recycler, queue.Queue)
    assert not c.recycler.empty()
    source2 = c.recycler.get()
    assert source2 is source
    c.set_recycler_capacity(0)
    assert c.recycler is None


def test_frame_flagspace(basic_source, populated_source):
    assert basic_source.frame_flagspace is None
    assert populated_source.frame_flagspace is not None


def test_channel_flagspace(basic_source, populated_source):
    assert basic_source.channel_flagspace is None
    assert populated_source.channel_flagspace is not None


def test_signal_mode(populated_source):
    assert populated_source.signal_mode.name == 'TOTAL_POWER'


def test_exclude_samples(populated_source):
    flags = populated_source.frame_flagspace.flags
    assert flags.BAD_DATA & populated_source.exclude_samples != 0


def test_logging_id(basic_source):
    assert basic_source.logging_id == 'model'


def test_n_scans(basic_source, populated_source):
    assert basic_source.n_scans == 0
    assert populated_source.n_scans == 1


def test_configuration(basic_source):
    source = basic_source.copy()
    assert isinstance(source.configuration, Configuration)
    source.info = None
    assert source.configuration is None


def test_has_option(basic_source):
    source = basic_source
    assert source.has_option('projection')
    assert not source.has_option('foo')


def test_source_option(basic_source):
    assert basic_source.source_option('foo') == 'source.foo'


def test_get_first_scan(basic_source, populated_source):
    assert basic_source.get_first_scan() is None
    assert populated_source.get_first_scan() is populated_source.scans[0]


def test_next_generation(basic_source):
    source = basic_source.copy()
    source.next_generation()
    assert source.generation == 1
    source.next_generation()
    assert source.generation == 2


def test_add_integration_time(basic_source):
    source = basic_source.copy()
    assert source.integration_time == 0
    source.add_integration_time(1.5 * units.Unit('second'))
    assert source.integration_time == 1.5 * units.Unit('second')


def test_set_info(basic_source):
    source = basic_source.copy()
    info = source.info
    source.info = None
    info.parent = None
    source.set_info(info)
    assert source.info is info
    assert info.parent is source


def test_add_process_brief(basic_source, capsys):
    source = basic_source.copy()
    assert source.process_brief is None
    source.add_process_brief('foo')
    assert isinstance(source.process_brief, list)
    assert source.process_brief == ['foo']
    source.add_process_brief(['bar', 'baz'])
    assert source.process_brief == ['foo', 'bar', 'baz']
    source.add_process_brief(1)
    assert "Received bad process brief message" in capsys.readouterr().err


def test_clear_process_brief(basic_source):
    source = basic_source.copy()
    source.process_brief = ['foo', 'bar']
    source.clear_process_brief()
    assert source.process_brief == []


def test_create_from(populated_scan, basic_source):
    scan = populated_scan.copy()
    scans = [scan]
    scan.info.astrometry.is_nonsidereal = True
    assert scan.is_nonsidereal
    source = basic_source.copy()
    source.configuration.parse_key_value('system', 'horizontal')
    assert source.configuration['system'] == 'horizontal'
    source.create_from(scans, assign_scans=True)
    assert source.scans == scans
    assert source.configuration['system'] == 'equatorial'
    assert scan.source_model is source


def test_assign_scans(populated_scan, basic_source):
    source = basic_source.copy()
    scan = populated_scan.copy()
    scans = [scan]
    scan.configuration.parse_key_value('jansky', '2.0')
    assert scan[0].gain == 1
    source.assign_scans(scans)
    assert scan[0].gain == 0.5
    assert scan.source_model is source


def test_assign_reduction(populated_scan, basic_source):
    source = basic_source.copy()
    scan = populated_scan.copy()
    reduction = source.reduction
    scans = [scan]
    reduction.scans = scans
    source.reduction = None
    source.info = None
    reduction.source = None
    reduction.configuration.objects.options['test'] = {'add': 'foo'}
    source.assign_reduction(reduction)
    assert source.reduction is reduction
    assert source.info is reduction.info
    assert source.scans == scans
    assert reduction.source is source
    assert reduction.configuration['foo']


def test_get_average_resolution(basic_source, populated_scan):
    source = basic_source.copy()
    scan1 = populated_scan.copy()
    scan2 = populated_scan.copy()
    scan2.channels = deepcopy(scan2.channels)
    scan1.weight = 2.0
    scan2.weight = 3.0
    i1, i2 = scan1[0].copy(), scan2[0].copy()
    i1.channels = scan1.channels
    i2.channels = scan2.channels
    scan1.integrations = [i1]
    scan2.integrations = [i2]
    i1.info.instrument.resolution = 4 * units.Unit('arcsec')
    i2.info.instrument.resolution = 9 * units.Unit('arcsec')
    i1.gain = 4.0
    i2.gain = 5.0
    source.scans = [scan1, scan2]
    assert np.isclose(source.get_average_resolution(),
                      7.8460657 * units.Unit('arcsec'), atol=1e-6)
    scan1.weight = 0
    scan2.weight = 0
    assert source.get_average_resolution() == 10 * units.Unit('arcsec')


def test_renew(populated_source):
    source = populated_source.copy()
    source.generation = 2
    source.integration_time = 2 * units.Unit('second')
    source.renew()
    assert source.generation == 0
    assert source.integration_time == 0


def test_reset_processing(basic_source):
    source = basic_source.copy()
    source.generation = 1
    source.integration_time = 1 * units.Unit('second')
    source.reset_processing()
    assert source.generation == 0 and source.integration_time == 0


def test_add_model(basic_source):
    s1 = basic_source.copy()
    s2 = basic_source.copy()
    s2.generation = 2
    s2.integration_time = 2 * units.Unit('second')
    s2.enable_level = False
    s2.enable_weighting = False
    s2.enable_bias = False
    assert s1.enable_level and s1.enable_weighting and s1.enable_bias
    s1.add_model(s2)
    assert s1.generation == 2
    assert s1.integration_time == 2 * units.Unit('second')
    assert not s1.enable_level
    assert not s1.enable_weighting
    assert not s1.enable_bias


def test_sync(populated_source):
    source = populated_source.copy()
    source.configuration.parse_key_value('source.coupling', 'True')
    source.sync()
    assert source.process_brief == ['(coupling)', '(sync)']
    assert source.scans[0].source_points == 1
    assert source.scans[0][0].source_generation == 1
    assert source.generation == 1
    source.process_brief = None
    source.configuration.parse_key_value('source.nosync', True)
    source.sync()
    assert source.process_brief is None


def test_sync_all_integrations(populated_source):
    source = populated_source.copy()
    assert source.synced_integrations == 0
    source.sync_all_integrations()
    assert source.synced_integrations == 1


def test_get_blanking_level(basic_source):
    source = basic_source.copy()
    source.configuration.parse_key_value('blank', '2.5')
    assert source.get_blanking_level() == 2.5
    del source.configuration['blank']
    assert np.isnan(source.get_blanking_level())


def test_get_clipping_level(basic_source):
    source = basic_source.copy()
    source.configuration.parse_key_value('clip', '10.0')
    assert source.get_clipping_level() == 10
    del source.configuration['clip']
    assert np.isnan(source.get_clipping_level())


def test_get_point_size(basic_source):
    source = basic_source.copy()
    assert source.get_point_size() == 10 * units.Unit('arcsec')


def test_get_source_size(basic_source):
    source = basic_source.copy()
    assert source.get_source_size() == 10 * units.Unit('arcsec')


def test_get_executor(basic_source):
    assert basic_source.get_executor() is None


def test_set_executor(basic_source):
    source = basic_source.copy()
    source.set_executor(None)


def test_get_parallel(basic_source):
    _ = basic_source.get_parallel()


def test_set_parallel(basic_source):
    source = basic_source.copy()
    source.set_parallel(2)


def test_no_parallel(basic_source):
    source = basic_source.copy()
    source.no_parallel()


def test_get_native_unit(basic_source):
    source = basic_source.copy()
    source.configuration.parse_key_value('dataunit', 'Kelvin')
    assert source.get_native_unit() == 1 * units.Unit('K')
    del source.configuration['dataunit']
    assert source.get_native_unit() == 1 * units.Unit('count')


def test_get_kelvin_unit(basic_source):
    source = basic_source.copy()
    k = source.get_kelvin_unit()
    assert k.unit == 'Kelvin'
    assert np.isnan(k)


def test_get_canonical_source_name(basic_source):
    assert basic_source.get_canonical_source_name() == 'test'

    class S2(FunctionalSourceModel):
        def get_source_name(self):
            return 'foo*bar\tbaz'

    s = S2(basic_source.info)
    assert s.get_canonical_source_name() == 'foo_bar_baz'


def test_get_default_core_name(populated_source):
    source = populated_source.copy()
    scan1 = source.scans[0]
    scan2 = deepcopy(scan1)
    scan1.info.observation.obs_id = 'Simulation.1'
    scan2.info.observation.obs_id = 'Simulation.2'
    source.scans.append(scan2)
    assert source.get_default_core_name() == 'test.Simulation.1-Simulation.2'
    scan1.mjd += 1
    assert source.get_default_core_name() == 'test.Simulation.2-Simulation.1'


def test_check_pixel_count(populated_source):
    source = populated_source.copy()
    integration = populated_source.scans[0][0]
    integration.configuration.parse_key_value('mappingpixels', '500')
    assert not source.check_pixel_count(integration)
    assert integration.comments[-1] == '(!ch)'
    del integration.configuration['mappingpixels']
    integration.configuration.parse_key_value('mappingfraction', '1.1')
    assert not source.check_pixel_count(integration)
    assert integration.comments[-1] == '(!ch%)'
    del integration.configuration['mappingfraction']
    assert source.check_pixel_count(integration)


def test_get_ascii_header(populated_source):
    source = populated_source.copy()
    lines = source.get_ascii_header().splitlines()
    assert lines[0].startswith('# SOFSCAN version')
    assert lines[1] == '# Instrument: example'
    assert lines[2] == '# Object: test'
    assert lines[3].startswith('# Equatorial:')
    assert lines[4] == '# Scans: Simulation.1'
    source.scans = None
    lines = source.get_ascii_header().splitlines()
    assert lines[-1] == '# Scans: '


def test_get_table_entry(basic_source):
    assert basic_source.get_table_entry('foo') is None


def test_parse_header(basic_source):
    header = fits.Header()
    header['INSTRUME'] = 'HAWC+'
    source = basic_source.copy()
    source.parse_header(header)
    assert isinstance(source.info, HawcPlusInfo)


def test_edit_header(populated_source):
    source = populated_source.copy()
    header = fits.Header()
    source.edit_header(header)
    assert 'DATE' in header
    assert header['SCANS'] == 1
    assert header['INTEGRTN'] == 0
    assert 'INSTRUME' in header
    #assert 'COUNTRY' in header
    assert 'HISTORY' in header


def test_add_scan_hdus_to(populated_source):
    source = populated_source.copy()
    hdul = fits.HDUList()
    source.configuration.parse_key_value('write.scandata', 'True')
    source.add_scan_hdus_to(hdul)
    assert len(hdul) == 2
    assert isinstance(hdul[0], fits.PrimaryHDU)  # automatic
    assert isinstance(hdul[1], fits.BinTableHDU)
    assert hdul[1].header['EXTNAME'] == 'Scan-Simulation.1'
    hdul = fits.HDUList()
    source.configuration.parse_key_value('write.scandata', 'False')
    source.add_scan_hdus_to(hdul)
    assert len(hdul) == 0
    source.configuration.parse_key_value('write.scandata', 'True')
    source.scans = None
    source.add_scan_hdus_to(hdul)
    assert len(hdul) == 0


def test_post_process_scan(populated_source):
    source = populated_source.copy()
    source.post_process_scan(source.scans[0])  # Doesn't do anything


def test_suggestions(populated_source, capsys):
    source = populated_source.copy()
    source.suggestions()
    out = capsys.readouterr().out
    assert "Please consult the README" in out
    assert "Check the console output" in out
    source.generation = 1
    source.suggestions()
    out = capsys.readouterr().out
    assert len(out) == 0

    source.scans[0][0].configuration.parse_key_value('mappingpixels', '1000')
    source.suggestions()
    out = capsys.readouterr().out
    assert "Adjust 'mappingpixels'" in out


def test_suggest_make_valid():
    msg = FunctionalSourceModel.suggest_make_valid()
    assert "Check the console output" in msg


def test_is_scanning_problem_only(populated_source, capsys):
    source = populated_source.copy()
    assert source.is_scanning_problem_only()
    scan1 = source.scans[0]
    i1 = scan1[0]
    i1.configuration.parse_key_value('mappingpixels', '1000')
    i2 = deepcopy(i1)
    i1.filter_time_scale = 0.01 * units.Unit('second')
    scan1.integrations.append(i2)
    assert not source.is_scanning_problem_only()
    assert 'Low scanning speed' in capsys.readouterr().out


def test_troubleshoot_few_pixels(populated_source, capsys):
    source = populated_source.copy()
    assert source.troubleshoot_few_pixels()
    source.scans[0][0].configuration.parse_key_value('mappingpixels', '1000')
    assert not source.troubleshoot_few_pixels()
    assert "Reduce with 'bright'" in capsys.readouterr().out
    source.configuration.parse_key_value('faint', 'True')
    assert not source.troubleshoot_few_pixels()
    assert "instead of 'faint'" in capsys.readouterr().out
    source.configuration.parse_key_value('deep', 'True')
    assert not source.troubleshoot_few_pixels()
    assert "instead of 'deep'" in capsys.readouterr().out
