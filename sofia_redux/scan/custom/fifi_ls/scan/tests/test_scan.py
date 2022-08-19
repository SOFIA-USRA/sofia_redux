# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.custom.fifi_ls.info.info import FifiLsInfo
from sofia_redux.scan.custom.fifi_ls.info.astrometry import \
    FifiLsAstrometryInfo
from sofia_redux.scan.custom.fifi_ls.integration.integration import \
    FifiLsIntegration
from sofia_redux.scan.custom.fifi_ls.scan.scan import FifiLsScan


@pytest.fixture
def basic_scan(fifi_simulated_channels):
    scan = FifiLsScan(fifi_simulated_channels)
    scan.hdul = fits.HDUList()
    return scan


def test_init(fifi_simulated_channels):
    scan = FifiLsScan(fifi_simulated_channels)
    assert scan.prior_pipeline_step is None
    assert not scan.use_between_scans


def test_info(basic_scan):
    assert isinstance(basic_scan.info, FifiLsInfo)


def test_copy(basic_scan):
    scan = basic_scan
    scan_2 = scan.copy()
    assert isinstance(scan_2, FifiLsScan)
    assert scan is not scan_2


def test_astrometry(basic_scan):
    assert isinstance(basic_scan.astrometry, FifiLsAstrometryInfo)


def test_get_id(basic_scan):
    scan = basic_scan.copy()
    assert scan.get_id() == 'P_2021-12-06_FI_F999B00001'
    scan.configuration.parse_key_value('fifi_ls.uncorrected', 'True')
    assert scan.get_id() == 'P_2021-12-06_FI_F999B00001-uncor'


def test_get_integration_instance(basic_scan):
    assert isinstance(basic_scan.get_integration_instance(), FifiLsIntegration)


def test_get_first_integration(fifi_simulated_scan):
    scan = fifi_simulated_scan.copy()
    integration = scan[0]
    scan.integrations.append(integration.copy())
    i1 = scan.get_first_integration()
    assert i1 is integration


def test_get_last_integration(fifi_simulated_scan):
    scan = fifi_simulated_scan.copy()
    integration = scan[0]
    scan.integrations.append(integration.copy())
    i2 = scan.get_last_integration()
    assert isinstance(i2, FifiLsIntegration)
    assert i2 is not integration


def test_get_first_frame(fifi_simulated_scan):
    assert fifi_simulated_scan.get_first_frame().fixed_index == 0


def test_get_last_frame(fifi_simulated_scan):
    assert fifi_simulated_scan.get_last_frame().fixed_index > 0


def test_getitem(fifi_simulated_scan):
    assert isinstance(fifi_simulated_scan[0], FifiLsIntegration)


def test_edit_scan_header(fifi_simulated_scan):
    h = fits.Header()
    scan = fifi_simulated_scan.copy()
    scan.prior_pipeline_step = 'foo'
    scan.edit_scan_header(h)
    assert h['PROCLEVL'] == 'foo'


def test_add_integrations_from_hdul(fifi_simulated_scan, fifi_simulated_hdul):
    scan = fifi_simulated_scan.copy()
    scan.integrations = None
    scan.hdul = fifi_simulated_hdul
    scan.add_integrations_from_hdul(fifi_simulated_hdul)
    assert scan.integrations is not None and scan.size == 1


def test_validate(fifi_simulated_scan):
    scan = fifi_simulated_scan.copy()
    scan.configuration.parse_key_value('betweenscans', 'True')
    scan.validate()
    assert scan.use_between_scans


def test_get_point_size(fifi_simulated_scan):
    c = fifi_simulated_scan.get_point_size()
    assert np.isclose(c.x, 6.2 * units.Unit('arcsec'))
    assert np.isclose(c.y, 6.2 * units.Unit('arcsec'))
    assert np.isclose(c.z, 0.0563619565 * units.Unit('um'), atol=1e-4)

    class DummySource(object):
        @staticmethod
        def get_point_size():
            return Coordinate2D1([10 * units.Unit('arcsec'),
                                  11 * units.Unit('arcsec'),
                                  12 * units.Unit('um')])

    scan = fifi_simulated_scan.copy()
    scan.source_model = DummySource()
    c = scan.get_point_size()
    assert c.x == 10 * units.Unit('arcsec')
    assert c.y == 11 * units.Unit('arcsec')
    assert c.z == 12 * units.Unit('um')



