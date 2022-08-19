# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import pytest

from sofia_redux.scan.custom.example.integration.integration import \
    ExampleIntegration
from sofia_redux.scan.custom.example.scan.scan import ExampleScan
from sofia_redux.scan.reduction.reduction import Reduction


def test_init():
    reduction = Reduction('example')
    info = reduction.info
    channels = info.get_channels_instance()
    scan = ExampleScan(channels, reduction=reduction)
    assert scan.reduction is reduction


def test_referenced_attributes(initialized_scan):
    assert 'hdul' in initialized_scan.referenced_attributes


def test_get_integration_instance(initialized_scan, scan_file):
    hdul = fits.open(scan_file)
    scan = initialized_scan
    scan.hdul = hdul
    integration = scan.get_integration_instance()
    assert isinstance(integration, ExampleIntegration)
    hdul.close()


def test_read(initialized_scan, scan_file):
    scan = initialized_scan
    scan.read(scan_file)
    assert isinstance(scan.integrations[0], ExampleIntegration)
    assert scan.hdul is None
    hdul = fits.open(scan_file)
    scan.integrations = None
    scan.read(hdul)
    assert isinstance(scan.integrations[0], ExampleIntegration)
    assert scan.hdul is None


def test_close_fits(initialized_scan, scan_file):
    hdul = fits.open(scan_file)
    scan = initialized_scan
    scan.hdul = hdul
    scan.close_fits()
    with pytest.raises(ValueError) as err:  # Check it's closed
        _ = hdul[0].data
    assert "closed file" in str(err.value)
    assert scan.hdul is None
    scan.close_fits()
    assert scan.hdul is None


def test_read_hdul(initialized_scan, scan_file):
    hdul = fits.open(scan_file)
    scan = initialized_scan
    scan.hdul = hdul
    scan.read_hdul(hdul)
    assert isinstance(scan.integrations[0], ExampleIntegration)
    hdul.close()


def test_read_integration(populated_scan, scan_file):
    hdul = fits.open(scan_file)
    scan = populated_scan
    scan.hdul = hdul
    scan.read_integration(hdul)
    scan.close_fits()
    assert isinstance(scan.integrations[0], ExampleIntegration)
    assert len(scan.integrations) == 1


def test_get_id(populated_scan):
    assert populated_scan.get_id() == 'Simulation.1'
