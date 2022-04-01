# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np
import pytest

from sofia_redux.scan.custom.sofia.channels.channel_data.channel_data import \
    SofiaChannelData
from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo


class ChannelData(SofiaChannelData):  # pragma: no cover
    def read_channel_data_file(self, filename):
        pass

    def get_pixel_count(self):
        pass

    def get_pixels(self):
        pass

    def get_mapping_pixels(self, indices=None, name=None, keep_flag=None,
                           discard_flag=None, match_flag=None):
        pass

    def get_overlap_indices(self, radius):
        pass

    def get_overlap_distances(self, overlap_indices):
        pass


@pytest.fixture
def sofia_channels():
    info = HawcPlusInfo()
    info.read_configuration('default.cfg')
    channels = info.get_channels_instance()
    return channels


@pytest.fixture
def sofia_channel_data(sofia_channels):
    return ChannelData(channels=sofia_channels)


def test_init(sofia_channels):
    channels = sofia_channels.copy()
    info = ChannelData(channels=channels)
    assert info.channels is channels


def test_info(sofia_channel_data):
    arcsec = units.Unit('arcsec')
    hz = units.Unit('Hz')
    ud = units.dimensionless_unscaled
    data = sofia_channel_data.copy()
    data.fixed_index = np.arange(10)
    info = data.info
    info.instrument.angular_resolution = 5 * arcsec
    info.instrument.frequency = 10 * hz
    data.apply_info(info)
    assert np.allclose(data.frequency, np.full(10, 10) * hz)
    assert np.allclose(data.angular_resolution, np.full(10, 5) * arcsec)
    info.instrument.angular_resolution = 5.0
    info.instrument.frequency = 10.0
    data.apply_info(info)
    assert np.allclose(data.frequency, np.full(10, 10) * hz)
    assert np.allclose(data.angular_resolution,
                       np.full(10, 5) * units.Unit('radian'))
    info.instrument.angular_resolution = 5.0 * ud
    info.instrument.frequency = 10.0 * ud
    data.apply_info(info)
    assert np.allclose(data.frequency, np.full(10, 10) * hz)
    assert np.allclose(data.angular_resolution,
                       np.full(10, 5) * units.Unit('radian'))
