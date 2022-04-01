# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.scan.custom.example.info.observation import \
    ExampleObservationInfo


@pytest.fixture
def configured_observation(populated_scan):
    configuration = populated_scan.configuration.copy()
    info = ExampleObservationInfo()
    info.configuration = configuration.copy()
    return info


def test_init():
    info = ExampleObservationInfo()
    assert info.scan_id is None
    assert info.obs_id is None


def test_apply_configuration(configured_observation):
    info = ExampleObservationInfo()
    info.apply_configuration()
    assert info.scan_id is None
    assert info.obs_id is None
    info = configured_observation.copy()
    info.apply_configuration()
    assert info.scan_id == '1'
    assert info.obs_id == 'Simulation.1'


def test_get_table_entry():
    info = ExampleObservationInfo()
    info.obs_id = 1
    info.scan_id = 2
    assert info.get_table_entry('obsid') == 1
    assert info.get_table_entry('scanid') == 2
    assert info.get_table_entry('foo') is None
