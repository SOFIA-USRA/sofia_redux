# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.custom.fifi_ls.integration.integration import \
    FifiLsIntegration
from sofia_redux.scan.source_models.spectral_cube import SpectralCube


def test_init():
    integration = FifiLsIntegration()
    assert integration.scan is None


def test_scan_astrometry(fifi_simulated_integration):
    info = fifi_simulated_integration.scan_astrometry
    assert info is not None
    assert info is not fifi_simulated_integration.info.astrometry


def test_apply_configuration():
    integration = FifiLsIntegration()
    integration.apply_configuration()  # Does nothing
    assert integration.configuration is None


def test_read(fifi_simulated_integration, fifi_simulated_hdul):
    integration = fifi_simulated_integration.copy()
    integration.frames.data.fill(np.nan)
    integration.read(fifi_simulated_hdul)
    assert not np.isnan(integration.frames.data).all()


def test_validate(fifi_simulated_integration):
    integration = fifi_simulated_integration.copy()
    data = integration.frames.data
    data[:, 0] = 0
    integration.frames.data = data
    integration.validate()
    assert integration.channels.data.flag[0] != 0


def test_flag_zeroed_channels(fifi_simulated_integration):
    integration = fifi_simulated_integration.copy()
    assert not integration.channels.data.flag.any()
    integration.frames.data.fill(0)
    integration.flag_zeroed_channels()
    assert np.all(integration.channels.data.flag != 0)


def test_get_full_id(fifi_simulated_integration):
    scan_id = fifi_simulated_integration.get_full_id()
    assert scan_id == 'P_2021-12-06_FI_F999B00001'


def test_get_first_frame(fifi_simulated_integration):
    frame = fifi_simulated_integration.get_first_frame()
    assert frame.size == 1 and frame.valid
    assert frame.fixed_index == 0


def test_get_last_frame(fifi_simulated_integration):
    frame = fifi_simulated_integration.get_last_frame()
    assert frame.size == 1 and frame.valid
    assert frame.fixed_index == fifi_simulated_integration.size - 1


def test_get_crossing_time(fifi_simulated_integration):
    integration = fifi_simulated_integration.copy()
    source = SpectralCube(info=integration.info)
    integration.scan.source_model = source
    t = integration.get_crossing_time()
    assert isinstance(t, units.Quantity) and t > 0 and np.isfinite(t)
    integration.scan.source_model = None
    t = integration.get_crossing_time()
    assert isinstance(t, units.Quantity) and t > 0 and np.isfinite(t)



