# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest

from sofia_redux.scan.custom.hawc_plus.info.info import HawcPlusInfo


@pytest.fixture
def sofia_camera():
    # Just uses HAWC
    info = HawcPlusInfo()
    info.read_configuration('default.cfg')
    camera = info.get_channels_instance()
    camera.data.fixed_index = np.arange(5)
    camera.data.set_default_values()
    return camera


def test_read_pixel_data(sofia_camera, tmpdir):
    camera = sofia_camera.copy()
    row = '1.0 1.0 - 1.0 1.0 1.0 0 1 1 1'
    filename = str(tmpdir.mkdir('test_read_pixel_data').join(
        'channel_file.dat'))
    with open(filename, 'w') as f:
        print(row, file=f)

    camera.read_pixel_data(filename)
    assert filename in camera.info.configuration_files


def test_read_rcp(sofia_camera, tmpdir):
    camera = sofia_camera.copy()
    row = '0 1 2'
    filename = str(tmpdir.mkdir('test_read_pixel_data').join('rcp.dat'))
    with open(filename, 'w') as f:
        print(row, file=f)
    camera.read_rcp(filename)
    assert filename in camera.info.configuration_files
