# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import numpy as np

from sofia_redux.scan.channels.channel_data.single_color_channel_data \
    import SingleColorChannelData
from sofia_redux.scan.custom.example.channels.channel_group.channel_group \
    import ExampleChannelGroup


class SingleColorCheck(SingleColorChannelData):
    """An un-abstracted class for testing"""

    def __init__(self):
        super().__init__()

    def read_channel_data_file(self, filename):
        pass

    def geometric_cols(self):
        return 4

    def geometric_rows(self):
        return 3


class TestSingleColorChannelData(object):

    def test_init(self, populated_data):
        assert isinstance(populated_data, SingleColorChannelData)
        assert isinstance(SingleColorCheck(), SingleColorChannelData)

    def test_get_overlap_distances(self, populated_data, overlaps):
        distance, unit = populated_data.get_overlap_distances(overlaps)
        assert unit is units.arcsec
        assert distance.data.size == 24
        assert np.allclose(distance.data,
                           [2, 4, 6, 8,
                            2, 0, 2, 4, 6,
                            4, 2, 0, 2, 4,
                            6, 4, 2, 0, 2,
                            8, 6, 4, 2, 0])

    def test_calculate_overlap_values(self, populated_data, overlaps):
        distance, unit = populated_data.get_overlap_distances(overlaps)
        populated_data.calculate_overlap_values(distance, 2 * units.arcsec)
        assert populated_data.overlaps.size == 0
        distance.data[:] = 0
        populated_data.calculate_overlap_values(distance, 2 * units.arcsec)
        assert populated_data.overlaps.size == 0

    def test_get_pixel_count(self, populated_data):
        assert populated_data.get_pixel_count() == populated_data.size

    def test_get_pixels(self, populated_data):
        assert populated_data.get_pixels() is populated_data

    def test_get_mapping_pixels(self, populated_data):
        assert isinstance(populated_data.get_mapping_pixels(),
                          ExampleChannelGroup)

    def test_get_rcp_string(self, populated_data):
        rcp = populated_data.get_rcp_string()
        assert rcp.startswith('ch\t[Gpnt]\t[Gsky]ch\tdX\tdY')
        assert rcp.strip().endswith('120\t1.000\t1.000\t-1.000e+01\t-1.000e+01')
        assert len(rcp.split('\n')) == 123

    def test_get_overlap_indices(self, mocker, populated_data):
        # example instrument implements geometric_overlap_indices
        assert populated_data.get_overlap_indices(
            10 * units.arcsec).size == 6188

        # test generic implementation
        data = SingleColorCheck()
        data.fixed_index = np.arange(16)
        data.channels = populated_data.channels

        idx = data.get_overlap_indices(10 * units.arcsec)
        assert idx.size == 132
