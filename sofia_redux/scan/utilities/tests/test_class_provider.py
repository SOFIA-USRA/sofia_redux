# Licensed under a 3-clause BSD style license - see LICENSE.rst

from sofia_redux.scan.utilities.class_provider import (
    fix_instrument_name, get_instrument_module_path, to_class_name,
    to_module_name, get_instrument_path_and_class_name, get_class_for,
    channel_class_for, channel_data_class_for,
    channel_group_class_for, frames_class_for, frames_instance_for,
    info_class_for, get_integration_class, get_scan_class,
    default_modality_mode, get_grid_class, get_projection_class,
    get_simulated_source_class)
from sofia_redux.scan.channels.modality.correlated_modality import \
    CorrelatedModality


def test_fix_instrument_name():
    assert fix_instrument_name(None) == ''
    assert fix_instrument_name('HAWC+') == 'hawc_plus'
    assert fix_instrument_name(' AbCdEfG ') == 'abcdefg'


def test_get_instrument_module_path():
    assert get_instrument_module_path(
        'HAWC+') == 'sofia_redux.scan.custom.hawc_plus'
    assert get_instrument_module_path(
        'FOO') == 'sofia_redux.scan.custom.foo'


def test_to_class_name():
    assert to_class_name(None) == ''
    assert to_class_name('') == ''
    assert to_class_name('a') == 'A'
    assert to_class_name('some_class_path') == 'SomeClassPath'


def test_to_module_name():
    assert to_module_name(None) == ''
    assert to_module_name('') == ''
    assert to_module_name('A') == 'a'
    assert to_module_name('SomeClassPath') == 'some_class_path'


def test_get_instrument_path_and_class_name():
    module, name = get_instrument_path_and_class_name('HAWC+')
    assert module == 'sofia_redux.scan.custom.hawc_plus'
    assert name == 'HawcPlus'


def test_get_class_for():
    c = get_class_for('HAWC+', 'frames')
    assert c.__name__.endswith('HawcPlusFrames')
    c = get_class_for('HAWC+', 'channels.channel_data')
    assert c.__name__.endswith('HawcPlusChannelData')
    c = get_class_for('HAWC+', 'channels.channels', other_module='')
    assert c.__name__.endswith('HawcPlusChannels')
    c = get_class_for('HAWC+', 'channels', other_module='channels')
    assert c.__name__.endswith('HawcPlusChannels')


def test_channels_class_for():
    assert channel_class_for(
        'HAWC+').__name__.endswith('HawcPlusChannels')


def test_channel_data_class_for():
    assert channel_data_class_for(
        'HAWC+').__name__.endswith('HawcPlusChannelData')


def test_channel_group_class_for():
    assert channel_group_class_for(
        'HAWC+').__name__.endswith('HawcPlusChannelGroup')


def test_frames_class_for():
    assert frames_class_for(
        'HAWC+').__name__.endswith('HawcPlusFrames')


def test_frames_instance_for():
    assert frames_instance_for(
        'HAWC+').__class__.__name__.endswith('HawcPlusFrames')


def test_info_class_for():
    assert info_class_for(
        'HAWC+').__name__.endswith('HawcPlusInfo')


def test_get_integration_class():
    assert get_integration_class(
        'HAWC+').__name__.endswith('HawcPlusIntegration')


def test_get_scan_class():
    assert get_scan_class(
        'HAWC+').__name__.endswith('HawcPlusScan')


def test_get_default_modality_mode():
    modality_instance = CorrelatedModality()
    mode = default_modality_mode(modality_instance)
    assert mode.__name__.endswith('CorrelatedMode')
    mode = default_modality_mode(CorrelatedModality)
    assert mode.__name__.endswith('CorrelatedMode')
    mode = default_modality_mode(None)
    assert mode.__name__.endswith('Mode')


def test_get_grid_class():
    assert get_grid_class('') is None
    assert get_grid_class(None) is None
    grid_class = get_grid_class('spherical_grid')
    assert grid_class.__name__.endswith('SphericalGrid')
    grid_class = get_grid_class('grid_2d')
    assert grid_class.__name__.endswith('Grid2D')


def test_get_projection_class():
    assert get_projection_class('') is None
    assert get_projection_class(None) is None
    p_class = get_projection_class('gnomonic')
    assert p_class.__name__.endswith('GnomonicProjection')


def test_get_simulated_source_class():
    assert get_simulated_source_class('') is None
    assert get_simulated_source_class(None) is None
    s_class = get_simulated_source_class('single_gaussian')
    assert s_class.__name__.endswith('SingleGaussian')
