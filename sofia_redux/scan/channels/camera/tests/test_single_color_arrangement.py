# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from sofia_redux.scan.channels.camera.single_color_arrangement \
    import SingleColorArrangement
from sofia_redux.scan.custom.example.info.info import ExampleInfo


@pytest.fixture
def populated_camera(populated_data):
    camera = SingleColorArrangement(info=ExampleInfo())
    camera.data = populated_data
    camera.initialize()
    return camera


class TestSingleColorArrangement(object):

    def test_init(self, populated_camera):
        # non-abstract: can init directly
        assert isinstance(populated_camera, SingleColorArrangement)

    def test_get_pixels(self, populated_camera):
        assert populated_camera.get_pixel_count() == populated_camera.size
        assert populated_camera.get_pixels() is populated_camera.data

    def test_get_mapping_pixels(self, populated_camera):
        # set some flags
        populated_camera.data.flag[:4] = 1
        group = populated_camera.get_mapping_pixels()
        assert group.size == populated_camera.size
        group = populated_camera.get_mapping_pixels(keep_flag=1)
        assert group.size == 4
        group = populated_camera.get_mapping_pixels(match_flag=1)
        assert group.size == 4
        group = populated_camera.get_mapping_pixels(discard_flag=1)
        assert group.size == populated_camera.size - 4
