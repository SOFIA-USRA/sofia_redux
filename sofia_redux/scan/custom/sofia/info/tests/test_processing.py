# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.processing import SofiaProcessingInfo
from sofia_redux.scan.custom.sofia.flags.quality_flags import QualityFlags


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['PROCSTAT'] = 'LEVEL_2'
    h['HEADSTAT'] = 'MODIFIED'
    h['DATAQUAL'] = 'corrected'
    h['N_SPEC'] = 1
    h['PIPELINE'] = 'sofscan'
    h['PIPEVERS'] = 'v1'
    h['PRODTYPE'] = 'sofscan-IMAGE'
    h['FILEREV'] = 'a'
    h['ASSC_AOR'] = '01,02,03'
    h['ASSC_MSN'] = 'm1,m2,m3'
    h['ASSC_FRQ'] = '1.5,2.5,3.5'
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaProcessingInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_class():
    assert SofiaProcessingInfo.flagspace == QualityFlags
    for i in range(6):
        assert isinstance(SofiaProcessingInfo.process_level_comment[i], str)


def test_init():
    info = SofiaProcessingInfo()
    assert info.associated_aors is None
    assert info.associated_mission_ids is None
    assert info.associated_frequencies is None
    assert info.process_level is None
    assert info.header_status is None
    assert info.software_name is None
    assert info.software_full_version is None
    assert info.product_type is None
    assert info.revision is None
    assert info.n_spectra == -1
    assert info.quality_level.name == 'NOMINAL'
    assert info.quality is None


def test_log_id():
    assert SofiaProcessingInfo().log_id == 'proc'


def test_apply_configuration(sofia_configuration):
    info = SofiaProcessingInfo()
    info.apply_configuration()
    assert info.process_level is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.process_level == 'LEVEL_2'
    assert info.header_status == 'MODIFIED'
    assert info.quality == 'corrected'
    assert info.n_spectra == 1
    assert info.software_name == 'sofscan'
    assert info.software_full_version == 'v1'
    assert info.product_type == 'sofscan-IMAGE'
    assert info.revision == 'a'
    assert info.associated_aors == ['01', '02', '03']
    assert info.associated_mission_ids == ['m1', 'm2', 'm3']
    assert info.associated_frequencies == [1.5, 2.5, 3.5]
    assert info.quality_level.name == 'CORRECTED'
    del info.options['ASSC_AOR']
    del info.options['ASSC_MSN']
    del info.options['ASSC_FRQ']
    info.options['DATAQUAL'] = 'foo'
    info.apply_configuration()
    assert info.associated_aors is None
    assert info.associated_mission_ids is None
    assert info.associated_frequencies is None
    assert info.quality_level.name == 'NOMINAL'


def test_get_product_type(sofia_info):
    info = sofia_info
    assert info.get_product_type(0) == 'HEADER'
    assert info.get_product_type(1) == '1D'
    assert info.get_product_type(2) == 'IMAGE'
    assert info.get_product_type(3) == 'CUBE'
    assert info.get_product_type(4) == '4D'
    assert info.get_product_type(-1) == 'UNKNOWN'
    assert info.get_product_type('foo') == 'UNKNOWN'


def test_get_level_name(sofia_info):
    assert sofia_info.get_level_name(2) == 'LEVEL_2'


def test_get_comment(sofia_info):
    info = sofia_info
    assert info.get_comment(None) == 'Unknown processing level.'
    assert info.get_comment(QualityFlags.flags.FAIL
                            ) == 'Raw engineering/diagnostic data.'
    assert info.get_comment('level3') == 'Corrected/reduced science data.'
    assert info.get_comment(4.0) == 'Flux-calibrated science data.'
    assert info.get_comment(-1) == 'Invalid processing level: -1'


def test_get_processing(sofia_info):
    info = sofia_info.get_processing(True, 3, QualityFlags.flags.NOMINAL)
    assert info.process_level == 'LEVEL_3'
    assert info.product_type == 'sofscan-CUBE'
    assert info.quality == 'nominal'


def test_edit_header(sofia_info):
    h = fits.Header()
    sofia_info.edit_header(h)
    assert h['PROCSTAT'] == 'LEVEL_2'
    assert h.comments['PROCSTAT'] == 'Raw uncalibrated science data.'
    assert h['HEADSTAT'] == 'MODIFIED'
    assert h['PIPELINE'] == 'sofscan'
    assert h['PIPEVERS'] == 'v1'
    assert h['PRODTYPE'] == 'sofscan-IMAGE'
    assert h['FILEREV'] == 'a'
    assert h['DATAQUAL'] == 'corrected'
    assert h['N_SPEC'] == 1
    assert h['ASSC_AOR'] == '01, 02, 03'
    assert h['ASSC_MSN'] == 'm1, m2, m3'
    assert h['ASSC_FRQ'] == '1.5, 2.5, 3.5'


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('q').name == 'CORRECTED'
    assert info.get_table_entry('nspec') == 1
    assert info.get_table_entry('quality') == 'corrected'
    assert info.get_table_entry('level') == 'LEVEL_2'
    assert info.get_table_entry('stat') == 'MODIFIED'
    assert info.get_table_entry('product') == 'sofscan-IMAGE'
    assert info.get_table_entry('foo') is None
