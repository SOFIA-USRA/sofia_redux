# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units
from astropy.io import fits
import numpy as np
import pytest

from sofia_redux.scan.custom.example.channels.channels import ExampleChannels


@pytest.fixture
def example_channels(populated_integration):
    return populated_integration.channels


def test_init(populated_integration):
    integration = populated_integration
    info = integration.info
    channels = ExampleChannels(parent=integration, info=info, size=100,
                               name='foo')
    assert channels.n_store_channels == 100
    assert channels.info.name == 'foo'
    assert channels.parent is integration
    assert channels.info is info


def test_copy(example_channels):
    channels = example_channels.copy()
    assert channels.divisions.keys() == example_channels.divisions.keys()


def test_init_divisions(example_channels):
    channels = example_channels.copy()
    channels.divisions = {}
    channels.init_divisions()
    assert 'mux' in channels.divisions
    assert 'bias' in channels.divisions
    mux = channels.divisions['mux']
    assert np.allclose(mux.groups[0].mux, 0)
    assert np.allclose(mux.groups[1].mux, 1)


def test_init_modalities(example_channels):
    channels = example_channels.copy()
    channels.modalities = {}
    channels.init_modalities()
    mux = channels.modalities['mux']
    assert mux.id == 'm'
    bias = channels.modalities['bias']
    assert bias.id == 'b'
    assert np.allclose(bias.modes[2].channel_group.bias_line, 2)


def test_read_data(example_channels):
    hdul = fits.HDUList()
    example_channels.read_data(hdul)  # Does nothing


def test_get_si_pixel_size(example_channels):
    channels = example_channels.copy()
    pixel_size = channels.get_si_pixel_size()
    assert pixel_size.x == 2 * units.Unit('arcsec')
    assert pixel_size.y == 2 * units.Unit('arcsec')


def test_load_channel_data(example_channels, capsys, tmpdir):
    path = tmpdir.mkdir('load_channel_data')
    wiring_file = str(path.join('wiring.dat'))
    with open(wiring_file, 'w') as f:
        print('hello', file=f)
    channels = example_channels.copy()
    c = channels.configuration
    c.parse_key_value('pixeldata', 'auto')
    c.parse_key_value('wiring', 'foo')
    channels.load_channel_data()
    captured = capsys.readouterr()
    assert "Initializing channel data" in captured.out
    assert "Cannot read wiring data" in captured.err
    c.parse_key_value('pixeldata', 'foo')
    c.parse_key_value('wiring', wiring_file)
    channels.load_channel_data()
    captured = capsys.readouterr()
    assert "Cannot read pixel data" in captured.err

    pixel_file = str(path.join('pixel.dat'))
    header = '# This is irrelevant'
    l1 = '1 2 g 3 4 5 6 7 8'
    l2 = '2 3 - 4 5 6 7 8 9'
    lines = [header, l1, l2]
    with open(pixel_file, 'w') as f:
        for line in lines:
            print(line, file=f)
    c.parse_key_value('pixeldata', pixel_file)
    channels.load_channel_data()
    assert channels.data.flag[85] == 8
    assert channels.data.flag[97] == 0
