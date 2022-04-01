# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.custom.example.integration.integration import \
    ExampleIntegration


@pytest.fixture
def hdul_scan(populated_scan, scan_file):
    scan = populated_scan
    scan.hdul = fits.open(scan_file)
    return scan


def test_init(hdul_scan):
    integration = ExampleIntegration(scan=hdul_scan)
    assert integration.scan is hdul_scan


def test_copy(populated_integration):
    integration = populated_integration.copy()
    assert np.allclose(integration.frames.data,
                       populated_integration.frames.data, equal_nan=True)


def test_read(populated_integration, scan_file):
    integration = populated_integration
    hdul = fits.open(scan_file)
    integration.frames.data = None
    integration.read(hdul)
    assert integration.frames.data.shape == (1100, 121)
    hdul.close()
    hdul = fits.HDUList()
    integration.frames.data = None
    integration.read(hdul)
    assert integration.frames.data is None
