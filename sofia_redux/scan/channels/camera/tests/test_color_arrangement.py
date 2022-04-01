# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
import pytest
import numpy as np

from sofia_redux.scan.channels.camera.color_arrangement import ColorArrangement
from sofia_redux.scan.custom.example.info.info import ExampleInfo


class ColorArrangementCheck(ColorArrangement):
    """An un-abstracted class for testing"""
    def read_channel_data_file(self, filename):
        pass

    def get_overlap_indices(self, radius):
        pass

    def get_overlap_distances(self, overlap_indices):
        pass

    def get_pixel_count(self):
        pass

    def get_pixels(self):
        return self.data

    def get_mapping_pixels(self, indices=None, name=None, keep_flag=None,
                           discard_flag=None, match_flag=None):
        pass


@pytest.fixture
def populated_camera(populated_data):
    camera = ColorArrangementCheck(info=ExampleInfo())
    camera.data = populated_data
    camera.initialize()
    return camera


class TestColorArrangement(object):

    def test_init(self):
        # can't init abstract class
        with pytest.raises(TypeError):
            ColorArrangement()

        # okay with abstract functions implemented
        ColorArrangementCheck()

    def test_apply_configuration(self, capsys, populated_camera):
        # set beam value
        populated_camera.configuration.set_option('beam', 10)
        populated_camera.apply_configuration()
        assert populated_camera.info.resolution == 10 * units.arcsec

        # set a missing alias
        populated_camera.configuration.set_option('beam', 'other')
        populated_camera.apply_configuration()
        assert populated_camera.info.resolution == 10 * units.arcsec
        assert 'Could not parse' in capsys.readouterr().err

        # set the alias
        populated_camera.configuration.set_option('other', 20)
        populated_camera.apply_configuration()
        assert populated_camera.info.resolution == 20 * units.arcsec

    def test_get_fwhm(self, populated_camera):
        channels = populated_camera
        nchannel = channels.size
        channels.data.resolution = np.arange(nchannel, dtype=float)
        channels.data.resolution[:2] = np.nan
        channels.data.resolution[-2:] = np.nan

        assert channels.get_min_beam_fwhm() == 2
        assert channels.get_max_beam_fwhm() == nchannel - 3
        assert channels.get_average_beam_fwhm() == (nchannel - 1) / 2
