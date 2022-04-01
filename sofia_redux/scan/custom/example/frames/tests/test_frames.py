# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pytest

from sofia_redux.scan.custom.example.frames.frames import ExampleFrames
from sofia_redux.scan.custom.example.flags.frame_flags import ExampleFrameFlags


@pytest.fixture
def example_frames(populated_integration):
    return populated_integration.frames


def test_class():
    assert ExampleFrames.flagspace == ExampleFrameFlags


def test_init():
    frames = ExampleFrames()
    assert frames.default_info is None


def test_copy(example_frames):
    frames = example_frames.copy()
    assert np.allclose(frames.data, example_frames.data, equal_nan=True)


def test_info(example_frames):
    frames = example_frames.copy()
    assert frames.info is not None
    frames.integration = None
    assert frames.info is None


def test_site(example_frames):
    frames = example_frames.copy()
    assert frames.site.latitude == 37.4089 * units.Unit('degree')


def test_read_hdu(example_frames, scan_file):
    frames = example_frames.copy()
    frames.valid[:] = False
    frames.data.fill(0)
    hdul = fits.open(scan_file)
    hdu = hdul[1]
    frames.read_hdu(hdu)
    assert not np.allclose(frames.data, 0)
    assert frames.valid.all()
    table = Table(hdu.data)
    table.remove_column('DAC')
    hdu = fits.BinTableHDU(data=table)
    hdul.close()
    frames.data.fill(0)
    frames.valid[:] = False
    frames.read_hdu(hdu)
    assert np.allclose(frames.data, 0)
    assert frames.valid.all()
