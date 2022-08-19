# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sofia_redux.scan.coordinate_systems.coordinate_2d import Coordinate2D
from sofia_redux.scan.coordinate_systems.coordinate_2d1 import Coordinate2D1
from sofia_redux.scan.custom.fifi_ls.channels.channels import FifiLsChannels
from sofia_redux.scan.custom.fifi_ls.info.info import FifiLsInfo
from sofia_redux.scan.custom.fifi_ls.info.detector_array import \
    FifiLsDetectorArrayInfo


@pytest.fixture
def fifi_basic_channels():
    info = FifiLsInfo()
    channels = FifiLsChannels(info=info)
    return channels


def test_init():
    info = FifiLsInfo()
    channels = FifiLsChannels(info=info)
    assert channels.n_store_channels == 400


def test_copy(fifi_basic_channels):
    channels = fifi_basic_channels
    channels_2 = channels.copy()
    assert isinstance(channels_2, FifiLsChannels)
    assert channels is not channels_2


def test_detector(fifi_basic_channels):
    assert isinstance(fifi_basic_channels.detector, FifiLsDetectorArrayInfo)


def test_pixel_sizes(fifi_basic_channels):
    channels = fifi_basic_channels.copy()
    assert channels.pixel_sizes.is_nan()
    assert channels.pixel_sizes.unit == 'arcsec'


def test_init_divisions(fifi_simulated_channels):
    channels = fifi_simulated_channels.copy()
    channels.divisions = None
    channels.init_divisions()
    divisions = channels.divisions
    for key in ['spexels', 'spaxels', 'rows', 'cols']:
        assert key in divisions


def test_init_modalities(fifi_simulated_channels):
    channels = fifi_simulated_channels.copy()
    del channels.divisions['spexels']
    channels.modalities = None
    channels.init_modalities()
    for key in ['spaxels', 'rows', 'cols']:
        assert key in channels.modalities
    assert 'spexels' not in channels.modalities


def test_load_channel_data(fifi_basic_channels):
    fifi_basic_channels.load_channel_data()  # should not do anything


def test_set_nominal_pixel_positions(fifi_simulated_channels):
    channels = fifi_simulated_channels.copy()
    channels.data.position = None
    channels.set_nominal_pixel_positions()
    assert isinstance(channels.data.position, Coordinate2D)
    assert channels.data.position.shape == (400,)


def test_max_pixels(fifi_basic_channels):
    assert fifi_basic_channels.max_pixels() == 400


def test_read_data(fifi_simulated_channels, fifi_simulated_hdul):
    channels = fifi_simulated_channels.copy()
    channels.data.position.zero()
    channels.read_data(fifi_simulated_hdul)
    assert not channels.data.position.is_null().all()


def test_get_si_pixel_size(fifi_basic_channels):
    channels = fifi_basic_channels.copy()
    assert isinstance(channels.get_si_pixel_size(), Coordinate2D)


def test_write_flat_field(fifi_simulated_channels, tmpdir):
    channels = fifi_simulated_channels.copy()
    filename = str(tmpdir.mkdir('test_write_flat_field').join(
        'flatfield.fits'))
    channels.write_flat_field(filename, include_nonlinear=True)
    hdul = fits.open(filename)
    assert 'CHANNEL GAIN' in hdul
    assert 'BAD PIXEL MASK' in hdul
    assert 'CHANNEL NONLINEARITY' in hdul


def test_add_hdu(fifi_simulated_channels):
    channels = fifi_simulated_channels.copy()
    hdul = fits.HDUList()
    hdu = fits.ImageHDU(data=np.zeros(10))
    channels.add_hdu(hdul, hdu, 'foo')
    assert hdu.header['EXTNAME'] == 'foo'


def test_calculate_overlaps(fifi_simulated_channels):
    channels = fifi_simulated_channels.copy()
    channels.data.overlaps = None
    channels.overlap_point_size = None
    channels.calculate_overlaps()
    assert isinstance(channels.overlap_point_size, Coordinate2D1)
    assert np.isclose(channels.overlap_point_size.x,
                      6.2 * units.Unit('arcsec'))
    assert np.isclose(channels.overlap_point_size.y,
                      6.2 * units.Unit('arcsec'))
    assert np.isclose(channels.overlap_point_size.z,
                      0.05636196 * units.Unit('um'), atol=1e-6)
    assert isinstance(channels.data.overlaps, csr_matrix)

    # Check not recalculated if point size is the same as before
    channels.data.overlaps = None
    channels.calculate_overlaps()
    assert channels.data.overlaps is None
