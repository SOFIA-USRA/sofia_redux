# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.configuration.configuration import Configuration
from sofia_redux.scan.custom.sofia.info.observation import SofiaObservationInfo


@pytest.fixture
def sofia_header():
    h = fits.Header()
    h['DATASRC'] = 'astro'
    h['OBSTYPE'] = 'object'
    h['SRCTYPE'] = 'point_source'
    h['KWDICT'] = 'DCS_SI_01_E'
    h['OBS_ID'] = '2016-12-14_HA_F358-90_0083_3-057'
    h['IMAGEID'] = 'img1'
    h['AOT_ID'] = '0083_3'
    h['AOR_ID'] = '90_0083_3'
    h['FILEGPID'] = '1234'
    h['FILEGP_R'] = 'r'
    h['FILEGP_B'] = 'b'
    h['OBJECT'] = 'uranus'
    return h


@pytest.fixture
def sofia_configuration(sofia_header):
    c = Configuration()
    c.read_configuration('default.cfg')
    c.read_fits(sofia_header)
    return c


@pytest.fixture
def sofia_info(sofia_configuration):
    info = SofiaObservationInfo()
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    return info


def test_init():
    info = SofiaObservationInfo()
    assert info.data_source is None
    assert info.obs_type is None
    assert info.source_type is None
    assert info.dictionary_version is None
    assert info.obs_id is None
    assert info.image_id is None
    assert info.aot_id is None
    assert info.aor_id is None
    assert info.file_group_id is None
    assert info.red_group_id is None
    assert info.blue_group_id is None
    assert not info.is_primary_obs_id


def test_apply_configuration(sofia_configuration):
    info = SofiaObservationInfo()
    info.apply_configuration()
    assert info.data_source is None
    info.configuration = sofia_configuration.copy()
    info.apply_configuration()
    assert info.data_source == 'astro'
    assert info.obs_type == 'object'
    assert info.source_type == 'point_source'
    assert info.dictionary_version == 'DCS_SI_01_E'
    assert info.obs_id == '2016-12-14_HA_F358-90_0083_3-057'
    assert info.image_id == 'img1'
    assert info.aot_id == '0083_3'
    assert info.aor_id == '90_0083_3'
    assert info.file_group_id == '1234'
    assert info.red_group_id == 'r'
    assert info.blue_group_id == 'b'
    assert info.source_name == 'uranus'


def test_edit_header(sofia_info, sofia_header):
    h = fits.Header()
    sofia_info.edit_header(h)
    for key, value in sofia_header.items():
        assert h[key] == value


def test_is_aor_valid(sofia_info):
    assert sofia_info.is_aor_valid()


def test_get_table_entry(sofia_info):
    info = sofia_info
    assert info.get_table_entry('aor') == '90_0083_3'
    assert info.get_table_entry('aot') == '0083_3'
    assert info.get_table_entry('obsid') == '2016-12-14_HA_F358-90_0083_3-057'
    assert info.get_table_entry('src') == 'astro'
    assert info.get_table_entry('dict') == 'DCS_SI_01_E'
    assert info.get_table_entry('fgid') == '1234'
    assert info.get_table_entry('imgid') == 'img1'
    assert info.get_table_entry('obj') == 'uranus'
    assert info.get_table_entry('objtype') == 'point_source'
    assert info.get_table_entry('foo') is None
