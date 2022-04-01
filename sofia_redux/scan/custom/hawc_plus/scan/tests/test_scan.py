# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.custom.hawc_plus.integration.integration import \
    HawcPlusIntegration
from sofia_redux.scan.custom.hawc_plus.scan.scan import HawcPlusScan


arcsec = units.Unit('arcsec')
um = units.Unit('um')


@pytest.fixture
def small_scan(small_integration):
    return small_integration.scan


def test_init(hawc_plus_channels):
    scan = HawcPlusScan(hawc_plus_channels)
    assert scan.prior_pipeline_step is None
    assert not scan.use_between_scans


def test_transit_tolerance(no_data_scan):
    scan = no_data_scan.copy()
    assert np.isclose(scan.transit_tolerance, np.nan * arcsec, equal_nan=True)
    scan.info.chopping.transit_tolerance = 5 * arcsec
    assert scan.transit_tolerance == 5 * arcsec


def test_focus_t_offset(no_data_scan):
    scan = no_data_scan.copy()
    assert np.isclose(scan.focus_t_offset, np.nan * um, equal_nan=True)
    scan.info.telescope.focus_t_offset = 5 * um
    assert scan.focus_t_offset == 5 * um


def test_info(no_data_scan):
    assert no_data_scan.info.name == 'hawc_plus'


def test_copy(no_data_scan):
    scan = no_data_scan.copy()
    assert scan is not no_data_scan
    assert scan.channels is no_data_scan.channels


def test_astrometry(no_data_scan):
    assert no_data_scan.astrometry.date == '2016-12-14'


def test_get_integration_instance(no_data_scan):
    scan = no_data_scan.copy()
    scan.hdul = fits.HDUList()
    integration = scan.get_integration_instance()
    assert isinstance(integration, HawcPlusIntegration)
    assert integration.scan is scan


def test_get_first_integration(no_data_scan):
    scan = no_data_scan.copy()
    scan.integrations = [1, 2, 3]
    assert scan.get_first_integration() == 1


def test_get_last_integration(no_data_scan):
    scan = no_data_scan.copy()
    scan.integrations = [1, 2, 3]
    assert scan.get_last_integration() == 3


def test_get_first_frame(small_scan):
    assert small_scan.get_first_frame().fixed_index == 0


def test_get_last_frame(small_scan):
    assert small_scan.get_last_frame().fixed_index == 9


def test_getitem(small_scan):
    assert small_scan[0] is small_scan.integrations[0]


def test_edit_scan_header(small_scan):
    scan = small_scan.copy()
    scan.prior_pipeline_step = 'The last step'
    scan.validate()
    h = fits.Header()
    scan.edit_scan_header(h)
    assert h['PROCLEVL'] == 'The last step'


def test_add_integrations_from_hdul(no_data_scan, full_hdu):
    hdul = fits.HDUList()
    hdul.append(full_hdu)
    scan = no_data_scan.copy()
    scan.hdul = hdul
    scan.integrations = None
    assert scan.size == 0
    scan.add_integrations_from_hdul(hdul)
    assert scan.size == 1


def test_validate(small_scan):
    scan = small_scan.copy()
    scan.configuration.parse_key_value('chopper.tolerance', '4.0')
    for integration in scan.integrations:
        integration.is_valid = False
    scan.is_nonsidereal = True
    scan.validate()
    assert not scan.is_nonsidereal
    assert scan.transit_tolerance == 4 * arcsec


def test_get_nominal_pointing_offset(small_scan):
    scan = small_scan.copy()
    native_pointing = Coordinate2D([1, 2], unit='arcsec')
    scan.configuration.parse_key_value('pointing', '3,4')
    offset = scan.get_nominal_pointing_offset(native_pointing)
    assert np.allclose(offset.coordinates, [168.613, 142.470] * arcsec,
                       atol=1e-3)


def test_get_table_entry(no_data_scan):
    scan = no_data_scan.copy()
    scan.info.telescope.focus_t_offset = 5 * um
    assert scan.get_table_entry('hawc.dfoc') == 5 * um
    assert scan.get_table_entry('foo') is None
